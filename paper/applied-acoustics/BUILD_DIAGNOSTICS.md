# Build diagnostics

Last checked: 2026-06-17

## Result

BasicTeX is installed locally for this user account and the manuscript can be
compiled on this machine.

## Toolchain status

Installed TeX root:

```text
/Users/zkr/usr/local/texlive/2026basic
```

Binary directory:

```text
/Users/zkr/usr/local/texlive/2026basic/bin/universal-darwin
```

Available required tools:

- `latexmk`
- `pdflatex`
- `kpsewhich`
- `xelatex`
- `lualatex`

Missing recommended tool:

- `biber`

The paper uses BibTeX through `elsarticle-num.bst`, so `biber` is not required
for the current build.

## Installed TeX packages

Additional packages installed after BasicTeX:

- `latexmk`
- `elsarticle`

The TeX Live repository is set to:

```text
https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet
```

## Verification

With the BasicTeX binary directory prepended to `PATH`, the LaTeX doctor reports
`existing-usable` and the TeX Live smoke test passes.

The project build command succeeds:

```bash
make compile
```

Current generated PDF:

```text
main.pdf
```

The current PDF build includes the synced final deterministic Conformer result
files and the updated Conformer table entries.

## Submission-check status

`make check` currently reports zero blockers. The only remaining warning is that
bundled Tectonic is not on `PATH`; this is not a blocker because BasicTeX is the
verified build toolchain.

Current verification command:

```bash
make compile
make check
```
