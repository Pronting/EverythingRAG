import { Children, useEffect, useId, useRef, useState, type ReactElement, type CSSProperties } from "react";

interface OptionProps { value: string; children: string; disabled?: boolean }
interface Props {
  value: string;
  onChange: (event: { target: { value: string } }) => void;
  "aria-label": string;
  children: ReactElement<OptionProps>[];
}

/** Theme-aware listbox; focus stays on the trigger for keyboard and screen-reader use. */
export default function SettingsSelect({ value, onChange, children, "aria-label": label }: Props) {
  const options = (Children.toArray(children) as ReactElement<OptionProps>[]).map((item) => item.props);
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [position, setPosition] = useState<CSSProperties>({});
  const selected = options.findIndex((option) => option.value === value);

  const show = () => {
    const rect = trigger.current!.getBoundingClientRect();
    const below = window.innerHeight - rect.bottom - 16;
    const above = rect.top - 16;
    const upwards = below < 200 && above > below;
    setPosition({ left: rect.left, width: rect.width, maxHeight: Math.min(240, upwards ? above : below),
      ...(upwards ? { bottom: window.innerHeight - rect.top + 6 } : { top: rect.bottom + 6 }) });
    setActive(selected >= 0 ? selected : options.findIndex((option) => !option.disabled));
    setOpen(true);
  };
  const choose = (index: number) => {
    if (!options[index] || options[index].disabled) return;
    onChange({ target: { value: options[index].value } });
    setOpen(false);
  };
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    /**
     * 只在「用户主动滚动」时关闭菜单。
     *
     * 不能用 scroll 事件：设置页正文（.settings-body）在聚焦触发按钮时会被浏览器
     * 程序化滚动几像素把按钮带进视野，这种滚动不是用户意图，却会把刚打开的菜单关掉
     * （长设置页表现为「点开就闪退」）。wheel / touchmove 只由真实用户输入产生。
     */
    const userScroll = (event: Event) => {
      const target = event.target;
      if (target instanceof Element && target.closest(".settings-select-menu") !== null) return;
      setOpen(false);
    };
    const close = () => setOpen(false);
    document.addEventListener("pointerdown", outside);
    window.addEventListener("resize", close);
    window.addEventListener("wheel", userScroll, { capture: true, passive: true });
    window.addEventListener("touchmove", userScroll, { capture: true, passive: true });
    return () => {
      document.removeEventListener("pointerdown", outside);
      window.removeEventListener("resize", close);
      window.removeEventListener("wheel", userScroll, { capture: true });
      window.removeEventListener("touchmove", userScroll, { capture: true });
    };
  }, [open]);
  useEffect(() => {
    if (!open) return;
    // 只滚动菜单自身：scrollIntoView 会连带滚动祖先滚动容器（如 .settings-body），
    // 其 scroll 事件不在菜单内，会立刻触发下面的「滚动即关闭」而把菜单关掉。
    const container = menu.current;
    if (container === null) return;
    const option = document.getElementById(`${id}-${active}`);
    if (option === null) return;
    const containerRect = container.getBoundingClientRect();
    const optionRect = option.getBoundingClientRect();
    if (optionRect.top < containerRect.top) {
      container.scrollTop -= containerRect.top - optionRect.top;
    } else if (optionRect.bottom > containerRect.bottom) {
      container.scrollTop += optionRect.bottom - containerRect.bottom;
    }
  }, [active, open, id]);

  return <div className="settings-select" ref={root}>
    <button type="button" className="settings-select-trigger" ref={trigger} role="combobox"
      aria-label={label} aria-expanded={open} aria-haspopup="listbox" aria-controls={open ? id : undefined}
      aria-activedescendant={open ? `${id}-${active}` : undefined}
      onClick={() => open ? setOpen(false) : show()} onBlur={() => setOpen(false)}
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) { event.preventDefault(); event.stopPropagation(); setOpen(false); return; }
        if (event.key === "Tab") { setOpen(false); return; }
        if (["ArrowDown", "ArrowUp", "Home", "End", "Enter", " "].includes(event.key)) {
          event.preventDefault();
          if (!open) { show(); return; }
          if (event.key === "Enter" || event.key === " ") { choose(active); return; }
          const enabled = options.map((option, index) => option.disabled ? -1 : index).filter((index) => index >= 0);
          const current = enabled.indexOf(active);
          setActive(event.key === "Home" ? enabled[0] : event.key === "End" ? enabled[enabled.length - 1] : enabled[(current + (event.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length]);
        }
      }}>
      <span>{options[selected]?.children ?? "请选择"}</span>
      <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </button>
    {open && <div className="settings-select-menu" id={id} role="listbox" aria-label={label} style={position} ref={menu}>
      {options.map((option, index) => <div key={option.value} id={`${id}-${index}`} role="option" aria-selected={option.value === value} aria-disabled={option.disabled || undefined}
        className={`settings-select-option${index === active ? " highlighted" : ""}`} onPointerMove={() => !option.disabled && setActive(index)}
        onPointerDown={(event) => event.preventDefault()} onClick={(event) => { event.preventDefault(); choose(index); }}>
        <span>{option.children}</span><span aria-hidden="true">{option.value === value ? "✓" : ""}</span>
      </div>)}
    </div>}
  </div>;
}
