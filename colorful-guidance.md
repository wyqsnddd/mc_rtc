# Guidance for reports based on `optimization-based-control.tex`

## 1. Purpose and authority

Use this guide when creating a LaTeX report that must follow
`cvae-development/reactive-control/optimization-based-control.tex`. The reference
report is authoritative for the document class, shared inputs, title page,
header, section hierarchy, mathematical notation, callouts, citation style,
figures, tables, bibliography, and appendices.

The relevant sources, in descending order of authority, are:

1. `reactive-control/optimization-based-control.tex` for report-level structure
   and usage examples.
2. `commands/configurations.tex` for semantic notation and presentation macros.
3. `commands/format-configurations.tex` for packages, page layout, headers,
   footers, colors, lists, tables, figures, and hyperlinks.
4. `commands/theorem-configurations.tex` for theorem-like environments.
5. `commands/yuquanTitle.sty` for `\project`, `\summary`, the title page, and
   teal section styling.
6. `commands/ref.bib` for shared bibliography entries.

All paths in that list are relative to `cvae-development/`. The reference
repository pins the `commands` submodule at commit
`7cdf487da0291d3f71929ac60ef31695859ffbb7`. Use the pinned submodule when exact
rendering is required; a newer `commands` checkout may rename or consolidate
macros.

Here, “strictly follow” means preserving the report's deliberate layout,
notation, structure, and writing patterns. It does not mean copying accidental
defects such as duplicate labels, repeated `\appendix` commands, typographical
errors, or content placed after `\end{document}`.

## 2. Directory and dependency contract

Place a new report one directory below `cvae-development/`, preferably beside
the reference report:

```text
cvae-development/
├── commands/                         # pinned shared LaTeX submodule
│   ├── configurations.tex
│   ├── format-configurations.tex
│   ├── theorem-configurations.tex
│   ├── yuquanTitle.sty
│   └── ref.bib
└── reactive-control/
    ├── optimization-based-control.tex # reference only; do not edit for a new report
    ├── <new-report>.tex
    └── <report-local-figure>.png
```

This depth is important: the literal path `../commands/...` in the preamble and
bibliography must resolve to `cvae-development/commands/...`. Keep figures used
by the report in the report directory, as the reference does, and include them
by filename. Do not add a different `\graphicspath` convention when strict
matching is required.

Before authoring or building, initialize the nested repository's submodules:

```sh
git -C cvae-development submodule update --init --recursive
```

Then verify that the exact dependencies exist:

```sh
test -f cvae-development/commands/format-configurations.tex
test -f cvae-development/commands/configurations.tex
test -f cvae-development/commands/theorem-configurations.tex
test -f cvae-development/commands/yuquanTitle.sty
test -f cvae-development/commands/ref.bib
```

Do not modify `commands/` or `figure/`, or advance their submodule pointers, as
part of an ordinary report-writing task.

## 3. Required document shell

Start from the following shell. Preserve the document class, option, package
and input order, title mechanism, and bibliography style. Replace only the
placeholders and body content.

```tex
\documentclass[10pt]{article}
% %%%%%%%%%%%%%%%%%%%%%%%%%% Table set up

\usepackage{tabu}
\input{../commands/format-configurations.tex}
\input{../commands/configurations.tex}
\input{../commands/theorem-configurations.tex}
\usepackage{../commands/yuquanTitle}


%%%%%%%%%%%%%%%%%%%%%%%%%%% -------------------------------------------


\project{\tinytf <Full report title>}
\author{
  <Author name> \\
  <Affiliation> \\
  <City and country>\\
  \vspace{5pt}
  \texttt{<email-address>}\vspace{30pt} \\
}
\summary{
  <Two or three sentences stating the scope, covered material, and practical
  purpose of the report.>
}

\rhead{Robotics-X \colorbox{teal}{\textcolor{white}{\quad \bf <Short running title>}}}
%%%%%%%%%%%%%%%%%%%%%%%%%%% -------------------------------------------


\begin{document}

\maketitle

\section{Introduction}
\label{sec:introduction}

<Motivation, context, scope, roadmap, and contribution.>

\section{<First technical block>}
\label{sec:first-block}

<Technical content.>

\section{<Synthesis or unified formulation>}
\label{sec:synthesis}

<Combine the earlier building blocks into the main formulation.>

\section{<Methods, solvers, or implementation choices>}
\label{sec:methods}

<Compare relevant alternatives and discuss tradeoffs.>

\section{Future works}

\begin{itemize}
  \item <Concrete open problem or planned extension.>
\end{itemize}

% Bibliography
% -----------------------------------------------------------------
\bibliography{../commands/ref}
\bibliographystyle{unsrtnat}

\appendix
\section{<Appendix title>}
\label{app:<appendix-key>}

<Supporting derivation, algorithm, figure, or implementation detail.>

\end{document}
```

The reference intentionally uses its custom title page instead of standard
`\title{...}` and does not enable an abstract or table of contents. Unless the
task explicitly asks for them, keep `abstract`, `\tableofcontents`, and their
associated page breaks absent. Do not add a separate `\title` command.

The source does not set `\date`; `yuquanTitle.sty` therefore renders LaTeX's
current date in the “Updated on” field. Set `\date{...}` only when the requested
report needs a reproducible or explicitly supplied date.

## 4. Content architecture

The reference is a technical tutorial. Its content progresses from physical
and mathematical building blocks to a unified optimization problem, then to
solver choices and reusable applications:

1. **Introduction** — motivation, historical context, limitations of earlier
   methods, scope, and a section-by-section roadmap.
2. **Constraints** — joint constraints first, then contact constraints; simple
   constraints precede viability and complementarity refinements.
3. **Balance constraints** — definitions and simplified models precede the
   individual CoM, ZMP, DCM, and angular-momentum constraints.
4. **Task formulation** — error dynamics are introduced by control level, then
   reused in different task types.
5. **QP formulation** — previously defined tasks and constraints are assembled
   into one optimization program.
6. **Task or constraint activation** — transition behavior is handled after the
   static formulation has been established.
7. **Solvers** — algorithms are compared by computational properties,
   limitations, and available implementations.
8. **Usual tasks** — the generic machinery is applied to common control tasks.
9. **Future works** — unresolved issues are listed briefly and concretely.
10. **Bibliography and appendices** — citations precede supporting algorithms,
    figures, or implementation details.

For another optimization-based robot-control report, retain this sequence and
rename only headings that genuinely do not apply. For a different technical
topic, preserve the same logical roles:

```text
motivation
  -> definitions and assumptions
  -> local models/constraints
  -> unified formulation
  -> numerical or implementation choices
  -> reusable examples
  -> open problems
  -> supporting appendices
```

Use no more heading depth than the reference:

- `\section` for major stages of the argument;
- `\subsection` for coherent technical families;
- `\subsubsection` for a specific constraint, task, model, or method;
- `\paragraph` for a tightly scoped variant within a subsubsection.

Do not use headings as substitutes for connective prose. Each section should
begin with a short orientation paragraph explaining why its components are
needed and how they relate.

## 5. Required writing pattern

For each technical concept, follow the recurring explanatory pattern in the
reference:

1. **Motivate the concept.** State the physical, numerical, or control-design
   issue in plain language.
2. **Declare assumptions and known quantities.** Identify frames, dimensions,
   sampling assumptions, contacts, bounds, and reference values before using
   them.
3. **Write the ideal relation.** Give the clearest physical or mathematical
   statement first.
4. **Derive the usable relation.** Show the substitutions or approximations
   needed for implementation.
5. **State the QP-ready form.** Put decision variables on the left and known
   terms or bounds on the right whenever practical.
6. **Interpret every displayed equation.** Immediately define symbols, explain
   the result, or state what the constraint guarantees.
7. **Connect forward and backward.** Cite the equation or section from which
   the result follows and the later formulation that consumes it.
8. **Isolate caveats.** Use `\shadowRemark{...}` for a numerical warning,
   modeling limitation, or implementation condition that should not interrupt
   the main derivation.

Write in a tutorial voice: direct, technical, and explanatory. Prefer “we
define,” “we impose,” “substituting,” “therefore,” and “where ... denotes ...”
when those phrases accurately describe the argument. Avoid unsupported claims,
undefined symbols, unexplained jumps between equations, and bare formula dumps.

The introduction must include all of the following:

- the practical role of the topic;
- the limitation or gap motivating the presented approach;
- representative citations for historical or state-of-the-art claims;
- the scope of the report;
- a roadmap using `\secRef{...}` rather than hard-coded section numbers.

## 6. Macro-first notation policy

### 6.1 Non-negotiable rule

Before typing a symbol directly, search `../commands/configurations.tex` for a
semantic macro. Reuse a shared macro whenever one exists, even when raw LaTeX
would be shorter. This keeps fonts, colors, frames, derivatives, superscripts,
subscripts, and sample indices consistent across the report.

Use this order of preference:

1. a domain-specific semantic macro such as `\jaccelerations`, `\com`, or
   `\wrench`;
2. a generic shared mathematical macro such as `\upperBound`, `\norm`, or
   `\innerP`;
3. ordinary LaTeX only when the shared configuration has no suitable command;
4. a small report-local macro only after confirming that the notation is
   missing and will be reused.

Useful searches from a report directory include:

```sh
rg -n '\\(newcommand|NewDocumentCommand).*\\jangles' ../commands/configurations.tex
rg -n '\\(newcommand|NewDocumentCommand).*\\(com|zmp|dcm|wrench|force)' ../commands/configurations.tex
rg -n 'QP|bound|constraint|error|Jacobian|momentum' ../commands/configurations.tex
```

Do not copy macro definitions out of `configurations.tex`. Do not locally
redefine a shared command to obtain a one-off appearance change.

### 6.2 Preferred substitutions

| Avoid hand-writing | Use instead |
|---|---|
| `\mathbf{\dot q}` | `\jvelocities` |
| `\mathbf{\ddot q}` | `\jaccelerations` |
| `\boldsymbol{\tau}` | `\jtorques` |
| `q_{\min}`, `q_{\max}` | `\lowerBound{\jangles}`, `\upperBound{\jangles}` |
| `x^{\mathrm{ref}}` | `\reff{x}` or the quantity macro's `ref` selector |
| `x^{\mathrm{des}}` | `\desired{x}` or the quantity macro's `des` selector |
| `x^\circ` | `\measured{x}` or the quantity macro's `mea` selector |
| `\mathbb{R}^n` | `\RRv{n}` |
| `\mathbb{R}^{m\times n}` | `\RRm{m}{n}` |
| `x^\top y` | `\innerP{x}{y}` |
| `A^\top` | `\transpose{A}` |
| `A^{-1}` | `\inverse{A}` |
| `\lVert x\rVert_2^2` | `\norm[two]{x}` |
| `[a,b]` | `\interval{a}{b}` |
| a raw set-builder expression | `\setDef{element}{condition}` |
| `\sum_{i=1}^{n}` | `\agg{i}{n}{...}` |
| `Sec.~\ref{...}` | `\secRef{...}` |
| `Fig.~\ref{...}` | `\figRef{...}` |
| `Remark~\ref{...}` | `\remarkRef{...}` |
| “state-of-the-art” | `\sota` |
| “mc_rtc” | `\mcrtc` |

The pinned command version used by the reference defines `\jvelocities` and
`\jaccelerations` explicitly. If those names are missing, the wrong macro
revision is probably being loaded; do not silently replace them until the
submodule version has been checked.

## 7. Macro catalog used by the reference

This is a usage guide, not a replacement for the macro definitions.

### 7.1 Document and presentation macros

| Macro | Purpose | Typical use |
|---|---|---|
| `\project{...}` | Full title rendered on the custom title page | `\project{\tinytf Optimization-based task-space robot control}` |
| `\summary{...}` | Title-page scope statement | Two or three complete sentences |
| `\tinytf` | Fixed title font used by the reference | Place at the start of `\project` |
| `\shadowRemark{...}` | Numbered, shadowed technical remark | Numerical caveat or modeling limitation |
| `\shadowProblem{title}{body}` | Shadowed problem statement | A formally posed feasibility or optimization problem |
| `\hlBlock{...}` | Teal inline label | Important item lead-in |
| `\infoBlock{...}` | Olive inline label | Method family or explanatory category |
| `\highlightblue{...}` | Blue equation/formulation heading | Constraint group |
| `\highlightgreen{...}` | Green equation/formulation heading | Contact or balance group |
| `\checkmark`, `\crossmark` | Feature-comparison markers | Solver table cells |

Use colored blocks sparingly. The reference employs them to organize dense
technical material, not to decorate ordinary prose.

### 7.2 Cross-reference macros

| Target | Label prefix | Reference form |
|---|---|---|
| Section | `sec:` | `\secRef{sec:tasks}` |
| Equation | `eq:` | `\eqref{eq:contact-cons}` |
| Figure | `fig:` | `\figRef{fig:prete-acc-bound}` |
| Table | `tab:` | `\tableRef{tab:solver-comparison}` |
| Remark | `re:` | `\remarkRef{re:friction-cone}` |
| Appendix | `app:` | `\appRef{app:derivation}` |

Place `\label` immediately after `\section`, `\subsection`, or `\caption`.
For equations created with `\quickEq`, pass the label as its first argument.
Every label must be unique. Use descriptive labels; do not copy generic labels
such as `tab:my_table` into a new report.

### 7.3 Equation and set macros

Use `\quickEq` for a single labeled equation:

```tex
\quickEq{eq:contact-constraint}{
  \jacobian \jaccelerations + \jacobian[dot] \jvelocities
  = \desired{\point[ddot]}.
}
```

Use an explicit `equation` plus `aligned` for a multi-line formulation:

```tex
\begin{equation}
  \label{eq:example-qp}
  \begin{aligned}
    \min_{\decisionVar} \quad &
      \agg{i}{n}{\weight_i \norm[two]{\function{i}(\decisionVar)}} \\
    \mbox{s.t.} \quad &
      \ieqC{i}\decisionVar \leq \ieqd{i}.
  \end{aligned}
\end{equation}
```

The reference's commonly reused mathematical helpers include:

| Macro | Meaning or result |
|---|---|
| `\quickEq{label}{body}` | A labeled `equation` environment |
| `\setDef{x}{condition}` | A set-builder definition |
| `\feasibleSet{x}`, `\viableSet{x}` | Feasible and viable sets |
| `\interval{a}{b}` | Closed interval |
| `\RR`, `\RRv{n}`, `\RRm{m}{n}` | Real scalar, vector, and matrix spaces |
| `\norm{x}` | Default norm |
| `\norm[two]{x}` | Squared Euclidean norm |
| `\norm[abs]{x}` | Scalar absolute value |
| `\twoNorm{x}` | Squared norm in the pinned command set |
| `\innerP{x}{y}` | Inner product `x^\top y` |
| `\transpose{x}`, `\inverse{x}` | Transpose and inverse |
| `\quadratic{x}{H}` | `\frac12 x^\top Hx` |
| `\agg{i}{n}{term}` | Sum from `i=1` to `n` |
| `\aggTwo{i}{a}{b}{term}` | Sum from an explicit lower to upper index |
| `\vectorTwo{a}{b}`, `\vectorThree{a}{b}{c}` | Column vectors |
| `\matrixTwo{A}{B}{C}{D}` | A two-by-two block matrix |
| `\twoCases{v_1}{c_1}{v_2}{c_2}` | Two-case expression |
| `\zeroVector` | Consistent zero-vector notation |

### 7.4 Bounds, time, and optimization macros

| Macro | Intended use |
|---|---|
| `\lowerBound{x}`, `\upperBound{x}` | Minimum and maximum of `x` |
| `\stateBound{x}` | `x_min <= x <= x_max` |
| `\stateAbsBound{x}` | Symmetric absolute bound |
| `\stateBoundConstraint{x}{expression}` | Bounds named after `x`, applied to an expression |
| `\initial{x}` | Initial value |
| `\current{x}`, `\previous{x}` | Current and previous control-cycle values |
| `\samplingPeriod` | Sampling period `\Delta t` |
| `\decisionVar` | QP decision vector |
| `\function{i}` | The `i`-th QP task functional |
| `\weight` | Task-weight symbol |
| `\ieqC{x}`, `\ieqd{x}` | Inequality matrix and vector notation |

When deriving a QP inequality, follow the reference's visual decomposition:

```tex
\begin{align}
  \current{\jaccelerations}
  &\leq
  \underbrace{
    \frac{\upperBound{\jvelocities}-\previous{\jvelocities}}
         {\samplingPeriod}
  }_{\ieqd{\upperBound{\jvelocities}}}, \\
  -\current{\jaccelerations}
  &\leq
  \underbrace{
    -\frac{\lowerBound{\jvelocities}-\previous{\jvelocities}}
          {\samplingPeriod}
  }_{\ieqd{\lowerBound{\jvelocities}}}.
\end{align}
```

Prefer a consistent `C\nu <= d` orientation for assembled QP inequalities.
Explain any sign reversal or substitution in the surrounding prose.

### 7.5 Structured quantity macros

Several quantities accept optional selectors. Preserve empty slots with `[]`
when setting a later selector.

The pinned `\jangles` signature is:

```tex
\jangles[derivative][status][coordinate-type][sample]
```

Examples:

```tex
\jangles                         % joint position
\jangles[dot]                    % joint velocity
\jangles[ddot]                   % joint acceleration
\jangles[][ref]                  % reference joint position
\jangles[][][actuated]           % actuated coordinates
\jangles[dot][mea][][previous]   % measured previous-sample velocity
```

The reference also uses the dedicated aliases `\jvelocities` and
`\jaccelerations` heavily. Follow the local surrounding style: use the aliases
for generic joint velocity and acceleration, and use `\jangles[...]` when
status, coordinate type, or sample selectors are needed.

The CoM, ZMP, and DCM families follow:

```tex
\com[derivative][status][component][sample]
\zmp[derivative][status][component][sample]
\dcm[derivative][status][component][sample]
```

Supported semantic selectors used by the shared configuration include:

- derivative: `dot`, `ddot`, or `dddot`;
- status: `ref`, `des`, `mea`, `est`, `min`, or `max`;
- component: `x`, `y`, `z`, or `plane`;
- sample: `time`, `current`, `next`, or `previous`.

Examples:

```tex
\com[ddot]              % CoM acceleration
\com[][ref][z]          % vertical reference CoM component
\zmp[][des][plane]      % desired planar ZMP
\dcm[][mea][][current]  % measured current DCM
```

Other structured families used in the reference include:

| Family | Typical form | Purpose |
|---|---|---|
| Jacobian | `\jacobian[dot][i]` | Derivative and semantic/index subscript |
| Point | `\point[ddot][des][i][current]` | Point derivative, status, index, and sample |
| Force | `\force[frame][point][status][component]` | Contact-force frame, application point, status, and axis |
| Torque | `\torque[frame][point][status][component]` | Contact-torque counterpart |
| Wrench | `\wrench[frame][point][status]` | Wrench frame, application point, and status |
| Angular momentum | `\angularMomentum[dot][frame][des][current]` | Momentum derivative, subscript/frame, status, and sample |
| Cone | `\cone[f][i]`, `\cone[cwc][i]` | Friction or contact-wrench cone for contact `i` |

If an optional-argument family is unfamiliar, inspect its definition before
guessing argument order.

### 7.6 Robot-control semantic macros

Use the established domain notation wherever applicable:

| Concept | Macro |
|---|---|
| Joint position, velocity, acceleration | `\jangles`, `\jvelocities`, `\jaccelerations` |
| Joint torque | `\jtorques` |
| Joint-space inertia | `\jsim` |
| Gravity and Coriolis term | `\gravityandcoriolis` |
| Jacobian | `\jacobian` |
| Contact force, torque, wrench | `\force`, `\torque`, `\wrench` |
| Number of sustained contacts | `\numContacts` |
| Friction coefficient | `\coef{f}` |
| Friction-cone generators | `\generator{\force}` |
| CoM, ZMP, DCM | `\com`, `\zmp`, `\dcm` |
| Textual “CoM” | `\comText` |
| Pendulum frequency and definition | `\pendulumC`, `\pendulumCDef` |
| Angular momentum | `\angularMomentum` |
| Centroidal momentum matrix | `\cmm` |
| Proportional, derivative, integral gains | `\pGain`, `\dGain`, `\iGain` |
| Error and its derivatives | `\error{x}`, `\errord{x}`, `\errordd{x}` |
| Reference and measured values | `\reff{x}`, `\measured{x}` |
| Inertial and centroidal frames | `\inertialFrame`, `\cmmFrame` |
| Geometric force transform | `\geometricFT{from}{to}` |

Do not mix equivalent raw symbols with these macros in the same report. For
example, if joint-space inertia is introduced as `\jsim`, do not later switch
to a hand-written `\mathbf{M}`.

## 8. Equations and derivations

### 8.1 Equation selection

- Use `\quickEq{...}{...}` for one labeled equation.
- Use `align` when several related lines need separate numbering.
- Use `align*` only when none of the lines will be referenced.
- Use `equation` with `aligned` for one logical formulation spanning several
  aligned lines.
- Use `\eqref{...}` for all numbered equation references.
- End displayed equations with punctuation when they are part of a sentence.

Avoid adding new `$$ ... $$` displays. The reference contains some legacy
instances, but the structured environments above are clearer and safer for
labels and alignment.

### 8.2 Derivation discipline

Every derivation must make these items explicit:

- the decision variable and its dimension when relevant;
- which quantities are measured, desired, fixed, or optimized;
- frame and application-point conventions for forces and wrenches;
- discretization method and sampling period for discrete constraints;
- assumptions used to simplify dynamics;
- whether a relation is exact, approximated, sufficient, or necessary;
- how the final expression enters the optimization problem.

Use underbraces with `\ieqC{...}` and `\ieqd{...}` when exposing a constraint's
matrix/vector structure materially improves comprehension. Do not add
underbraces to equations that are already obvious.

### 8.3 Formal callouts

Use `\shadowProblem` when the problem statement itself is important:

```tex
\shadowProblem{<Short problem name>}{
  Given <known data>, find <unknown quantity> such that <requirements>.
  \begin{equation}
    \label{eq:<problem-key>}
    \begin{aligned}
      <formal optimization problem>
    \end{aligned}
  \end{equation}
}
```

Use `\shadowRemark` for a localized warning or insight:

```tex
\shadowRemark{
  \label{re:<remark-key>}
  <Numerical, physical, or implementation caveat.>
}
```

Keep the main line of reasoning understandable without requiring the reader to
open an appendix or infer the purpose of a callout.

## 9. Citations and bibliography

The shared format loads `natbib` with numeric citations. Match the reference:

- use `\citet{key}` when authors are grammatical subjects;
- use `\cite{key}` for a parenthetical or supporting citation;
- use `\citet[Alg.~1]{key}` or another optional locator when pointing to a
  specific algorithm, equation, or section;
- place a nonbreaking space before a citation when it follows a term or author
  phrase, for example `methods~\cite{key}`;
- never write author names, years, or bibliography numbers manually when a BibTeX
  entry is available.

Search the shared bibliography before creating a report-local `.bib` file:

```sh
rg -n '<author|title-word|citation-key>' ../commands/ref.bib
```

Use exactly this order at the end of the main body:

```tex
\bibliography{../commands/ref}
\bibliographystyle{unsrtnat}
\appendix
```

If the task explicitly requires sources missing from `commands/ref.bib`, prefer
a small report-local bibliography and list both databases, for example
`\bibliography{../commands/ref,<new-report>}`. Do not edit the shared bibliography
unless the task explicitly includes a shared-reference update.

## 10. Lists, tables, figures, and code

### 10.1 Lists

Use `itemize` for unordered properties, options, and open problems. Use
`enumerate` when sequence or numbered cross-references matter. Indent nested
environment contents by two spaces:

```tex
\begin{itemize}
  \item \hlBlock{Error definition}: <explanation>.
  \item \hlBlock{Desired behavior}: <explanation>.
\end{itemize}
```

The reference customizes some enumerations with `\setlist`; keep such changes
local to the relevant list family and restore or replace them before unrelated
lists if necessary.

### 10.2 Tables

Use the reference table sequence:

```tex
\begin{table}[h]
  \centering
  \caption{<Descriptive caption>}
  \label{tab:<descriptive-key>}
  \begin{tabular}{|c|c|c|}
    \hline
    <Heading 1> & <Heading 2> & <Heading 3> \\
    \hline
    <Value> & \checkmark & \crossmark \\
    \hline
  \end{tabular}
\end{table}
```

Use `\makecell{...}` for intentional line breaks in narrow cells. Captions must
describe the comparison, and surrounding prose must explain the conclusion;
do not leave interpretation entirely to the table.

### 10.3 Figures

Keep report-specific bitmap figures beside the `.tex` file and follow:

```tex
\begin{figure}[htbp!]
  \centering
  \includegraphics[width=0.8\columnwidth]{<figure-file>.png}
  \caption{<What is shown and, when applicable, the source or algorithm.>}
  \label{fig:<descriptive-key>}
\end{figure}
```

Introduce and interpret every figure in the body with `\figRef{...}`. Use the
smallest legible width and preserve the aspect ratio. Do not copy generated
figures into the shared `figure/` submodule without explicit authorization.

### 10.4 Code listings

The shared format already loads `listings`. Configure the language immediately
before the relevant implementation material:

```tex
\lstset{
  language=c++,
}

\begin{lstlisting}
<short, relevant code fragment>
\end{lstlisting}
```

Keep listings short and explain why each fragment matters. Do not place working
notes or unfinished placeholders after `\end{document}`.

## 11. Source irregularities that must not be copied

The reference remains the visual and structural authority, but a new report
must correct these accidental issues:

- `eq:wholebody_qp` is used for more than one equation. Every new label must be
  unique.
- `\appendix` appears twice. A new report must call it exactly once.
- The reference contains material after `\end{document}`. LaTeX ignores it; all
  intended content must appear before the single final `\end{document}`.
- Some headings and prose contain spelling mistakes. Proofread new text rather
  than copying those mistakes.
- The reference has legacy `$$ ... $$` displays. Prefer `\quickEq`, `equation`,
  or `align` in new material.
- Commented-out draft blocks are historical notes, not required template
  content. Do not carry large stale comment blocks into a new report.

These corrections do not change the intended report style.

## 12. Authoring workflow

Follow this sequence for every new report:

1. Copy only the required document shell into a new, descriptive lowercase
   `.tex` filename. Do not duplicate or overwrite the reference report.
2. Draft the section outline using the content architecture above.
3. Make a notation inventory: states, derivatives, frames, references,
   measurements, bounds, decision variables, and gains.
4. Search `configurations.tex` for every inventory item and record the selected
   macro before writing equations.
5. Write the introduction and its roadmap after the section labels are stable.
6. Develop each concept with the motivate–define–derive–interpret pattern.
7. Assemble the unified formulation only after all constituent terms have been
   introduced and labeled.
8. Add comparisons, standard tasks, future work, bibliography, and appendices.
9. Build from the report directory and resolve every missing citation,
   undefined reference, duplicate label, and LaTeX error.
10. Review the generated PDF for title-page layout, header width, equation
    overflow, table width, figure readability, and page breaks.

## 13. Build and validation

Run LaTeX from the document directory so `../commands/` and local figures
resolve correctly:

```sh
cd cvae-development/reactive-control
pdflatex <new-report>.tex
bibtex <new-report>
pdflatex <new-report>.tex
pdflatex <new-report>.tex
```

Inspect serious warnings and errors:

```sh
rg -n 'LaTeX Error|undefined references|Citation.*undefined|multiply defined|Overfull \\hbox' <new-report>.log
```

The report is complete only when all of the following are true:

- the build exits successfully through the full BibTeX cycle;
- the title page, teal header/footer rules, and running header render correctly;
- every citation and cross-reference resolves;
- every label is unique and descriptive;
- all intended material appears before one `\end{document}`;
- notation is consistent and predominantly produced by shared macros;
- equations state assumptions and define every symbol;
- figures and tables are cited and interpreted in the prose;
- there are no serious overfull boxes or unreadable displays;
- the generated PDF has been reviewed visually;
- temporary `.aux`, `.bbl`, `.blg`, `.log`, and `.out` files are not committed;
- `commands/` and `figure/` submodule pointers remain unchanged unless their
  updates were explicitly requested.

## 14. Final compliance checklist

Before handing off a report, answer every item with “yes”:

- [ ] Does the file use `\documentclass[10pt]{article}`?
- [ ] Does it load `tabu`, the three shared configuration inputs, and
      `yuquanTitle` in the reference order?
- [ ] Are `\project`, `\author`, `\summary`, `\rhead`, and `\maketitle` present?
- [ ] Does the report follow the reference's tutorial progression?
- [ ] Does the introduction state motivation, scope, citations, and roadmap?
- [ ] Were shared macros searched before any notation was written manually?
- [ ] Are frames, statuses, bounds, and sample indices represented consistently?
- [ ] Does each displayed equation have context and an immediate interpretation?
- [ ] Are labeled single equations written with `\quickEq` where suitable?
- [ ] Do all sections, equations, figures, tables, remarks, and appendices use the
      prescribed label prefixes and reference commands?
- [ ] Are narrative citations written with `\citet` and supporting citations
      with `\cite`?
- [ ] Are the shared `unsrtnat` bibliography and its placement preserved?
- [ ] Is `\appendix` used at most once?
- [ ] Is there exactly one `\end{document}`, after all intended content?
- [ ] Has the full LaTeX–BibTeX–LaTeX–LaTeX build succeeded?
- [ ] Has the PDF been inspected, not merely compiled?
- [ ] Are generated build artifacts excluded from the change set?

