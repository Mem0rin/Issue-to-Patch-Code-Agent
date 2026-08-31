# src/code_agent/tool_registry.py
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import TypeAlias

from .kernel_types import ToolSpec

class InvalidToolArgumentsError(ValueError):
    """Tool arguments do not satisfy the runtime contract."""


ToolArgumentValidator: TypeAlias = Callable[
    [Mapping[str, object]],
    None,
]

ToolHandler: TypeAlias = Callable[
    [Mapping[str, object]],
    str,
]


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    validate_arguments: ToolArgumentValidator
    handler: ToolHandler


class DuplicateToolError(ValueError):
    """Two tools use the same public name."""


class UnknownToolError(LookupError):
    """A requested tool is not registered."""


class ToolRegistry:
    def __init__(
        self,
        definitions: Sequence[ToolDefinition],
    ) -> None:
        definitions_by_name: dict[
            str,
            ToolDefinition,
        ] = {}

        for definition in definitions:
            name = definition.spec.name

            # TODO 1：判断名称是否已经注册
            if name in definitions_by_name:
                raise DuplicateToolError(
                    f"duplicate tool name: {name}"
                )

            # TODO 2：把定义放进字典
            definitions_by_name[name] = definition

        self._definitions_by_name = definitions_by_name

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            definition.spec
            for definition
            in self._definitions_by_name.values()
        )

    def resolve(self, name: str) -> ToolDefinition:
        try:
            # TODO 3：通过名称取得 ToolDefinition
            return self._definitions_by_name[name]
        except KeyError as exc:
            raise UnknownToolError(
                f"unknown tool: {name}"
            ) from exc