export default function CommandCenter() {
  return (
    <div className="flex min-h-full flex-col bg-[var(--color-background)]">
      <header className="border-b border-[var(--color-border)] px-6 py-4">
        <h1 className="text-xl font-semibold tracking-tight">RD-Agent Terminal</h1>
        <p className="text-sm text-[var(--color-muted)]">Market view scaffold — Phase 1</p>
      </header>
      <main className="flex flex-1 items-center justify-center p-8 text-[var(--color-muted)]">
        Command Center loading...
      </main>
    </div>
  );
}
