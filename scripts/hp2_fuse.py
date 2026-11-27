"""Fuse a set of Harry Potter 2's VU1 programs into one C++ function (the native dispatcher, runtime/src/hp2_vu1*.cpp).

    python scripts/hp2_fuse.py <live VU folder kept with PS2_VU_LIVE_KEEP=1, of a fast-mode run> <out.inc> <function name> <entry=key> ...

The generated (vurecomp, fast mode) bodies of the programs are taken verbatim (kSticky = false where the body has the template), joined
with one set of register locals: each body's prologue becomes a re-entry block (the clamped shadows recomputed from the raw registers
and the flag pipeline restarted, as a fresh call of that program does after a hop wrote the registers back), and every way out that hops
to one of the set's entries jumps to that block. Any other way out writes every register back and returns the address the chain loop
goes on at; the end of the program (the E bit: r.tpc set) returns kEnded. The function takes the entry it starts at.
"""
import glob, re, sys

SRC, OUT, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
ENTRIES = {}
for a in sys.argv[4:]:
    e, k = a.split('=')
    ENTRIES[int(e, 16)] = k.upper()

text = {}
for f in glob.glob(SRC + '/work-*/src/*.cpp'):
    s = open(f, encoding='utf-8', errors='replace').read()
    for e, k in ENTRIES.items():
        if e in text:
            continue
        m = re.search(r'(template <bool kSticky>\n)?(PS2_VU_BODY|static inline __attribute__\(\(always_inline\)\)) void vu1_%s_b\(.*?\n\}\n' % k, s, re.S)
        if m:
            text[e] = m.group(0)
missing = ['%04X' % e for e in ENTRIES if e not in text]
if missing:
    sys.exit('missing bodies: %s' % missing)

decl_vf, decl_n, decl_i = set(), set(), set()
extra_decl, tables = set(), []
blocks = []
for e in sorted(ENTRIES):
    body = text[e]
    body = body.split('{\n', 1)[1].rsplit('\n}\n', 1)[0]
    tag = 'B%04X_' % e
    pro, rest, in_pro = [], [], True
    for l in body.split('\n'):
        if in_pro and l.startswith('    goto L_'):
            in_pro = False
        (pro if in_pro else rest).append(l)
    enter = []
    renames = []
    for l in pro:
        l = l.strip()
        if not l or l.startswith('(void)'):
            continue
        m = re.match(r'Reg128 (vf\d+|vacc) = r\.(vf\[\d+\]|acc);', l)
        if m:
            decl_vf.add((m.group(1), m.group(2)))
            continue
        m = re.match(r'v4u (n\w+) = (vshadow\(\w+\));', l)
        if m:
            decl_n.add(m.group(1))
            enter.append('%s = %s;' % (m.group(1), m.group(2)))
            continue
        m = re.match(r'std::uint16_t (i\d+) = r\.vi\[(\d+)\];', l)
        if m:
            decl_i.add((m.group(1), m.group(2)))
            continue
        if l.startswith('FlagPipeV fl;') or l.startswith('FlagPipe fl;'):
            enter.append('fl = FlagPipeV{};')
            if l.startswith('FlagPipe fl;'):
                raise SystemExit('%04X uses the full flag pipeline (reads the status): not fused' % e)
            continue
        if l.startswith('fl.start(r);'):
            enter.append('fl.start(r);')
            continue
        if l.startswith('bool pend = false'):
            enter.append('pend = false, pendj = false, ending = false;')
            continue
        if l.startswith('std::uint32_t jpc = 0;'):
            enter.append('jpc = 0;')
            continue
        if l.startswith('const std::uint32_t sh ='):
            continue
        # the branch delay of the integer registers and the pipeline state of path-dependent stalls (vurecomp_gen.cpp): state of one run
        if l.startswith('std::uint16_t bk = 0, next_bk = 0;'):
            extra_decl.add('std::uint16_t bk = 0, next_bk = 0;')
            enter.append('bk = 0, next_bk = 0;')
            continue
        if l.startswith('unsigned bi = 0;'):
            extra_decl.add('unsigned bi = 0;')
            enter.append('bi = 0;')
            continue
        m = re.match(r'unsigned st = (\d+)u;', l)
        if m:
            extra_decl.add('unsigned st = 0;')
            enter.append('st = %su;' % m.group(1))
            continue
        m = re.match(r'static constexpr std::uint8_t (k[se]_[0-9A-F_]+)\[\] = (.*);', l)
        if m:
            tables.append('    static constexpr std::uint8_t %s%s[] = %s;' % (tag, m.group(1), m.group(2)))
            renames.append(m.group(1))
            continue
        raise SystemExit('unexpected prologue line in %04X: %s' % (e, l))
    code = '\n'.join(rest).replace('kSticky', 'false')
    for name in renames:
        code = re.sub(r'\b%s\b' % name, tag + name, code)
    code = re.sub(r'\bL_([0-9A-F]{4})\b', tag + r'L_\1', code)
    code = code.replace('dispatch:', tag + 'dispatch:').replace('goto dispatch;', 'goto ' + tag + 'dispatch;')
    code = re.sub(r'\n[^\n]*?vhop\(hop, ([^;]*?)\);\n\s*return;', lambda m: '\n    { fl.finish(r); next = %s; goto route; }' % m.group(1), code)
    # the end of the program: "if (ending) { <syncs> r.tpc = X; return; }"
    code = re.sub(r'r\.tpc = ([^;]*);\s*return;', lambda m: 'fl.finish(r); r.tpc = %s; next = kEnded; goto route;' % m.group(1), code)
    if re.search(r'\bvhop\b|\breturn\b', code):
        raise SystemExit('an exit of %04X was not rewritten' % e)
    blocks.append((e, tag, enter, code))

allvf = sorted(decl_vf, key=lambda x: (x[0] == 'vacc', int(x[0][2:]) if x[0] != 'vacc' else 99))
alli = sorted(decl_i, key=lambda x: int(x[1]))
out = ['// Generated by scripts/hp2_fuse.py from vurecomp\'s fast-mode output (do not edit by hand): programs ' +
       ', '.join('%04X' % e for e in sorted(ENTRIES)),
       'std::uint32_t %s(Vu0Regs &__restrict__ r, std::uint8_t *mem, std::uint32_t entry) {' % NAME,
       '    const std::uint32_t sh = 0;', '    (void)sh;', '    VuDaz daz;', '    (void)daz;', '    FlagPipeV fl;',
       '    bool pend = false, pendj = false, ending = false;', '    std::uint32_t jpc = 0, next = 0;',
       '    (void)pend; (void)pendj; (void)ending; (void)jpc;']
for d in sorted(extra_decl):
    out.append('    ' + d)
out.extend(tables)
for name, src in allvf:
    out.append('    Reg128 %s = r.%s;' % (name, src))
for n in sorted(decl_n):
    out.append('    v4u %s{};' % n)
for name, idx in alli:
    out.append('    std::uint16_t %s = r.vi[%s];' % (name, idx))
out.append('    next = entry;')
out.append('    goto route_in;')
for e, tag, enter, code in blocks:
    out.append('%senter:' % tag)
    out.append('    ' + ' '.join(enter))
    out.append(code)
out.append('route:')
out.append('route_in:')
out.append('    switch (next) {')
for e, tag, enter, code in blocks:
    # the code at that address must be the code the block was made from (its key, this code generation): else the chain loop runs it
    out.append('    case 0x%04Xu: if (key_ok(0x%04Xu, 0x%sull)) goto %senter; break;' % (e, e, ENTRIES[e], tag))
out.append('    default: break;')
out.append('    }')
out.append('    ' + ' '.join('r.%s = %s;' % (src, name) for name, src in allvf) + ' ' + ' '.join('r.vi[%s] = %s;' % (idx, name) for name, idx in alli))
out.append('    return next;')
out.append('}')
# the entries and their keys, for the caller: whether an address is one of this function's programs (and which code it must hold)
out.append('std::uint64_t %s_key(std::uint32_t pc) {' % NAME)
out.append('    switch (pc) {')
for e in sorted(ENTRIES):
    out.append('    case 0x%04Xu: return 0x%sull;' % (e, ENTRIES[e]))
out.append('    default: return 0;')
out.append('    }')
out.append('}')
open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
print('fused %d programs into %s: %d lines, %d vector registers, %d integer registers' % (len(blocks), NAME, len(out), len(allvf), len(alli)))
