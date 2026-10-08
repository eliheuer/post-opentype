# The NeuralType viewer

Two programs show a NeuralType font: the web demo on elih.net
(`src/components/NeuralTypeDemo.tsx`) and the model view in Runebender's
proof strip (`src/application/platform/model.rs` and
`src/application/widgets/model_strip.rs`). This file is the contract
between them. Change it first, then change both programs in the same
session.

## 1. Shared code

Both programs call `neuraltype_core::field_line`:

- `build_field_line` lays out the words right to left on one baseline
  and applies the pulls.
- `marks` gives one node per caret index and one span per character.

Do not copy this code into a viewer. A change to layout or to node
placement goes into `field_line`, and then both viewers update their
engine: the web demo through `neuraltype-wasm`, Runebender through its
pinned `rev`.

## 2. Shared behavior

These must work the same way in both viewers.

| Thing | Behavior |
|---|---|
| Strand | The path the pen took: between two nodes of a word, through the middle of the stroke (`field_line::strand`); between words, a straight segment. |
| Node | One per caret index, always on the ink and mid-stroke, never on a dot. Where letters touch, at their join; where they do not, where the previous letter comes nearest the next. A node is hollow when it touches a word boundary (start, end, or beside a space). All nodes have one size. |
| Active node | The node at the caret. It is larger, filled, and has a rotating half-ring. |
| Click on ink | Moves the caret to the nearest caret index. |
| Drag from ink | Selects a range. Field fonts show the selection as a cloud, the union of the selected letters' fields, traced at a raised level. |
| Drag a node | A press on any node makes it the active node and drags it in one motion. It pulls that letter and the rest of its word; the join before it stretches. |
| Opening caret | One node in from the start, after the first letter. |
| Pulls | Belong to one text. A change to the text clears them. |

## 3. Shared colors

Each color has one name. The web demo uses the value in the table. Runebender
uses the theme role, so the color follows the editor's theme.

| Name | Web demo | Runebender role | Use |
|---|---|---|---|
| ground | `#0c0c0c` | proof strip ground | behind everything |
| ink | `#2aa35f` | proof ink | the letters |
| strand | `#ef4444` | `danger` | strand, nodes, caret |
| active | `#f97316` | `selection` | active node, cloud edge |
| ring | `#facc15` | `pointSelected` | the half-ring |
| cloud | `rgba(160,160,160,0.22)` | `previewFill` at 22% | selection fill |

Every mark has a 1 px ground-colored rim, so it stays visible on ink.

## 4. Where the viewers differ

These differences are allowed. Each one comes from the context.

| Web demo | Runebender |
|---|---|
| Teaches. The reader types in the canvas itself. | Edits. Text comes from the Neural section's Text field or from the open sample. |
| One font, chosen by the page. Controls for precision, tracer, strand and structure view. | A version list in the Neural section. No precision or tracer controls. |
| Fixed palette (section 3). | Theme palette, so it fits the gray and dark themes. |
| Fullscreen button. | The strip has the size that the editor layout gives it. |
| Shows the font's size and weight count. | Shows the version's score and epochs. |

When a new difference is necessary, add it to this table with its reason.
