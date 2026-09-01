# Agent Kernel

This context describes the language for one bounded Agent run and the records it produces so model input, runtime history, audit evidence, and future memory are not conflated.

## Language

**Run**:
One bounded attempt by the Agent Kernel to handle a task under a fixed set of resource and permission limits.
_Avoid_: Session, conversation

**Model Context**:
The exact information presented to one model call during a Run.
_Avoid_: Run History, memory

**Run History**:
The ordered messages, model actions, feedback, and tool results accumulated while a Run is active and available for constructing Model Context.
_Avoid_: Model Context, audit log

**Run Event**:
A structured fact about a meaningful state transition that occurred during a Run.
_Avoid_: Debug message

**Run Event Trace**:
The append-only ordered record of Run Events used to reconstruct and audit a Run without automatically becoming Model Context or future memory.
_Avoid_: Model Context, cross-run memory, ordinary debug log

**Cross-run Memory**:
Knowledge deliberately selected or derived from earlier Runs for possible use in a later Run.
_Avoid_: Run Event Trace, raw history
