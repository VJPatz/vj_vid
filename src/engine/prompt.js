/** Prompt scheduling — user's text kept verbatim across beats, only camera framing varies. */

export function buildPromptSchedule(prompt) {
  const scene = String(prompt ?? '').trim().replace(/\s+/g, ' ');
  if (!scene) return ['a serene empty landscape, cinematic'];
  return [
    `${scene}, wide establishing shot, cinematic, highly detailed`,
    `${scene}, three-quarter view, cinematic, highly detailed`,
    `${scene}, close detail, cinematic, highly detailed`,
  ];
}
