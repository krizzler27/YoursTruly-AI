export const store = {
  state: {
    selectedModel: null,
    isStreaming: false,
    currentConversationId: null,
    conversations: [],
    openHistoryMenuId: null,
    renamingId: null,
    pendingHistoryDeleteId: null,
    stagedFiles: new Map(),
    serverDocs: [],
    uploadingFiles: new Map(),
    docPollTimer: null,
    docPollStartedAt: 0,
    prevIndexedNames: new Set(),
    composerBlocked: false,
    contextCollapsed: false
  },
  _target: new EventTarget(),
  set(patch) {
    Object.assign(this.state, patch);
    this._target.dispatchEvent(new CustomEvent('change', { detail: patch }));
  },
  subscribe(cb) {
    const handler = (e) => cb(this.state, e.detail);
    this._target.addEventListener('change', handler);
    return () => this._target.removeEventListener('change', handler);
  }
};
