/**
 * Markdown heading utilities shared between the renderer and any
 * component that wants a table of contents (e.g. the Cookbook
 * provider guide's left rail). Keeping the slugify and extraction
 * logic in one place ensures anchor ids match between the two.
 */

export function slugifyHeading(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export interface MarkdownHeading {
  level: 2 | 3;
  text: string;
  id: string;
}

const HEADING_PATTERN = /^(#{2,3})\s+(.+?)\s*#*\s*$/gm;

/**
 * Extract H2 and H3 headings from a markdown source string. Skips
 * fenced code blocks so `# comment` inside ```` doesn't get picked up.
 */
export function extractHeadings(source: string): MarkdownHeading[] {
  // Strip fenced code blocks so `##` inside code doesn't pollute the TOC.
  const stripped = source.replace(/```[\s\S]*?```/g, "");
  const out: MarkdownHeading[] = [];
  let match: RegExpExecArray | null;
  HEADING_PATTERN.lastIndex = 0;
  while ((match = HEADING_PATTERN.exec(stripped)) !== null) {
    const level = match[1].length === 2 ? 2 : 3;
    const text = match[2].trim();
    out.push({ level, text, id: slugifyHeading(text) });
  }
  return out;
}

/**
 * Turn the flat MarkdownHeading list into text representing the H2
 * anchor targets only — what the Cookbook left rail uses.
 */
export function topLevelHeadings(
  source: string,
): ReadonlyArray<MarkdownHeading> {
  return extractHeadings(source).filter((h) => h.level === 2);
}
