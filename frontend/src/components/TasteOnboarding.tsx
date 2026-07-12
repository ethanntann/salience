import { FormEvent, useState } from "react";

type TasteKey =
  | "shotgun_one_pump"
  | "shotgun_kill"
  | "pistol_kill"
  | "automatic_kill"
  | "flick_shot"
  | "sniper_kill"
  | "no_scope"
  | "build_fight"
  | "fast_edit"
  | "clutch"
  | "multi_kill"
  | "victory"
  | "high_damage_hit"
  | "cleanup_kill"
  | "downed_finish"
  | "spray_kill"
  | "competitive_context"
  | "stationary_target";

const QUESTIONS: Array<{ key: TasteKey; label: string }> = [
  { key: "shotgun_kill", label: "Shotgun kills" },
  { key: "shotgun_one_pump", label: "One-pump shotgun kills" },
  { key: "pistol_kill", label: "Pistol kills" },
  { key: "automatic_kill", label: "Automatic rifle or SMG kills" },
  { key: "flick_shot", label: "Flick shots" },
  { key: "sniper_kill", label: "Sniper or Hunting Rifle kills" },
  { key: "no_scope", label: "True no-scope shots" },
  { key: "build_fight", label: "Build fights" },
  { key: "fast_edit", label: "Fast edits and mechanics" },
  { key: "clutch", label: "Clutch or last-teammate moments" },
  { key: "multi_kill", label: "Multi-kills" },
  { key: "victory", label: "Victory Royale endings" },
  { key: "high_damage_hit", label: "Big damage hits" },
  { key: "cleanup_kill", label: "Cleanup kills" },
  { key: "downed_finish", label: "Finishes on already-downed enemies" },
  { key: "spray_kill", label: "Spray kills" },
  { key: "competitive_context", label: "Ranked or tournament moments" },
  { key: "stationary_target", label: "Easy stationary targets" }
];

const OPTIONS = [
  { label: "Love", value: 2 },
  { label: "Like", value: 1 },
  { label: "Neutral", value: 0 },
  { label: "Avoid", value: -1 },
  { label: "Hate", value: -2 }
];

export function TasteOnboarding({
  onSave,
  onReset,
  busy
}: {
  onSave: (preferences: Record<string, number>) => Promise<void>;
  onReset: () => Promise<void>;
  busy: boolean;
}) {
  const [preferences, setPreferences] = useState<Record<string, number>>({
    cleanup_kill: -1,
    downed_finish: -2,
    shotgun_one_pump: 1,
    high_damage_hit: 1
  });

  async function submitPreferences(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSave(preferences);
  }

  async function resetPreferences() {
    setPreferences({});
    await onReset();
  }

  return (
    <section className="onboardingPanel">
      <div className="onboardingHeader">
        <div>
          <h2>Set your taste</h2>
          <p>Give the ranker a head start before you review a bunch of clips.</p>
        </div>
      </div>

      <form className="tasteGrid" onSubmit={submitPreferences}>
        {QUESTIONS.map((question) => (
          <label className="tasteQuestion" key={question.key}>
            <span>{question.label}</span>
            <select
              value={preferences[question.key] ?? 0}
              onChange={(event) =>
                setPreferences((current) => ({ ...current, [question.key]: Number(event.target.value) }))
              }
            >
              {OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        ))}
        <div className="tasteActions">
          <button disabled={busy}>Apply taste profile</button>
          <button className="ghost" type="button" disabled={busy} onClick={resetPreferences}>
            Reset taste
          </button>
        </div>
      </form>
    </section>
  );
}
