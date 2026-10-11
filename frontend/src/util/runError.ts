import type { AgentError } from "../types";

export function requiredActionText(error: Pick<AgentError, "required_action" | "resumable">): string {
  const action = error.required_action?.trim() ?? "";
  // 历史事件中的恢复接口转换为页面操作说明。
  if (/^POST\s+\/api\/workspaces\/.*\/runs\/.*\/resume\b/.test(action)) {
    return error.resumable ? "点击“继续执行”重试当前任务" : "请重新发送问题";
  }
  return action;
}
