import { useEffect, useRef } from "react";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  /** 确认按钮文案（如「确认删除」）。 */
  confirmLabel: string;
  /** 危险操作：确认按钮用警示色。 */
  danger?: boolean;
  busy?: boolean;
  /** 稳定测试钩子（E2E 按 testid 定位具体弹窗）。 */
  testId?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * 轻量确认弹窗（删除等不可撤销操作必须二次确认）。
 *
 * 打开时把焦点放到「取消」上并按 Esc 关闭，避免误触确认。
 */
export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  danger = false,
  busy = false,
  testId,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const cancelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    cancelRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div className="confirm-overlay" onPointerDown={onCancel}>
      <div
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        data-testid={testId}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <h3 className="confirm-dialog-title">{title}</h3>
        <p className="confirm-dialog-description">{description}</p>
        <div className="confirm-dialog-actions">
          <button type="button" className="confirm-cancel" ref={cancelRef} onClick={onCancel}>
            取消
          </button>
          <button
            type="button"
            className={danger ? "confirm-ok danger" : "confirm-ok"}
            disabled={busy}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
