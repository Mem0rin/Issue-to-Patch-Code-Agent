"""Store and consume one-shot tool approval requests."""

from dataclasses import dataclass

from .kernel_types import ToolCall


class DuplicateApprovalError(ValueError):
    """A pending approval already uses this ID."""


class UnknownApprovalError(LookupError):
    """The approval ID is unknown or already consumed."""


def _require_non_blank(
    value: str,
    field_name: str,
) -> None:
    if not value.strip():
        raise ValueError(
            f"{field_name} must not be blank"
        )


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    call: ToolCall
    reason: str

    def __post_init__(self) -> None:
        _require_non_blank(
            self.approval_id,
            "approval_id",
        )
        _require_non_blank(
            self.reason,
            "reason",
        )


@dataclass(frozen=True)
class ApprovalResponse:
    approval_id: str
    approved: bool

    def __post_init__(self) -> None:
        _require_non_blank(
            self.approval_id,
            "approval_id",
        )


@dataclass(frozen=True)
class ApprovalResolution:
    request: ApprovalRequest
    approved: bool


class PendingApprovals:
    def __init__(self) -> None:
        self._pending: dict[
            str,
            ApprovalRequest,
        ] = {}

    def add(self, request: ApprovalRequest) -> None:
        approval_id = request.approval_id

        # TODO 1：相同 ID 尚未处理时，拒绝覆盖
        if approval_id in self._pending:
            raise DuplicateApprovalError(
                f"duplicate approval: {approval_id}"
            )

        # TODO 2：保存请求
        self._pending[approval_id] = request

    def resolve(
        self,
        response: ApprovalResponse,
    ) -> ApprovalResolution:
        try:
            # TODO 3：取出并删除待确认请求
            request = self._pending.pop(response.approval_id)
        except KeyError as exc:
            raise UnknownApprovalError(
                "approval is unknown or already consumed"
            ) from exc

        return ApprovalResolution(
            request=request,
            approved=response.approved,
        )
