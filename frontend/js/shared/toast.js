import { ICONS } from './icons.js';

export function createToast(toastStack) {
  return function toast(msg, type = 'info', ttl) {
    if (!toastStack) return;
    const el = document.createElement('div');
    el.className = `toast-item ${type}`;
    const iconMap = { info: ICONS.info, success: ICONS.check, warning: ICONS.warn, error: ICONS.close };
    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.innerHTML = iconMap[type] || ICONS.info;
    const text = document.createElement('span');
    text.textContent = msg;
    text.style.flex = '1';
    text.style.minWidth = '0';
    const close = document.createElement('button');
    close.className = 'toast-close';
    close.type = 'button';
    close.setAttribute('aria-label', 'Dismiss');
    close.innerHTML = ICONS.close;
    close.addEventListener('click', () => dismiss());
    el.append(icon, text, close);
    toastStack.appendChild(el);
    const duration = ttl ?? (type === 'error' ? 6000 : type === 'warning' ? 5000 : 4000);
    let t = setTimeout(dismiss, duration);
    function dismiss() {
      clearTimeout(t);
      el.style.animation = 'toastOut 140ms ease forwards';
      setTimeout(() => el.remove(), 150);
    }
    el.addEventListener('mouseenter', () => clearTimeout(t));
    el.addEventListener('mouseleave', () => t = setTimeout(dismiss, 1200));
  };
}
