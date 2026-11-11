// Executes the pinned upstream formatter on synthetic inputs only. No API calls,
// real session files, agent loop, or replacement compression implementation.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const sourcePath = `${homedir()}/.agent-reach/research/ec2-bench-1fc4980/skills__cloud-advisor__plugin__index.ts`;
const source = readFileSync(sourcePath, 'utf8');
const sha = createHash('sha256').update(source).digest('hex');
assert.equal(sha, '8876761fb369cfc2dfc600a5089935727ef091448294cbfd1bf5b4800e67befd');
const start = source.indexOf('function buildTranscriptText(): string {');
const end = source.indexOf('async function callCloudAdvisor(', start);
assert(start >= 0 && end > start);
const code = stripTypeScriptTypes(source.slice(start, end), { mode: 'strip' });

function format(messages) {
  const serialized = messages.map(message => JSON.stringify({ type: 'message', message })).join('\n');
  const fixture = '/synthetic-fixture-only/chat.jsonl';
  const context = {
    CHAT_JSONL: fixture,
    existsSync(path) { assert.equal(path, fixture); return true; },
    readFileSync(path) { assert.equal(path, fixture); return serialized; },
  };
  vm.runInNewContext(`${code}\nresult = buildTranscriptText();`, context, { timeout: 1000 });
  return context.result;
}
function lcp(a, b) {
  let i = 0;
  while (i < Math.min(a.length, b.length) && a[i] === b[i]) i++;
  return i;
}
const initial = [{ role: 'user', content: 'Analyze the public fixture.' },
                 { role: 'assistant', content: 'I will inspect its columns.' }];
const first = format(initial);
const appended = format([...initial, { role: 'toolResult', toolName: 'read', content: 'column_a,column_b\n1,2' }]);
assert(appended.startsWith(first));
const longMessages = Array.from({ length: 210 }, (_, i) => ({ role: 'assistant', content: `step-${i}:` + 'x'.repeat(1000) }));
const longBefore = format(longMessages);
const longAfter = format([...longMessages, { role: 'assistant', content: 'new-observation:' + 'y'.repeat(1000) }]);
const longPrefix = lcp(longBefore, longAfter);
assert(longPrefix >= 30000 && longPrefix < 30100);
const imageOnly = format([{ role: 'user', content: [{ type: 'image', data: 'synthetic-image-sentinel' }] }]);
assert.equal(imageOnly, '');
const toolTrim = format([{ role: 'toolResult', toolName: 'read', content: 'a'.repeat(4000) }]);
assert(toolTrim.includes('truncated 1000 chars'));
assert(toolTrim.length < 3100);

const result = {
  upstream_commit: '1fc4980707d7944bc1e047c2e1844423a9f811af', source_sha256: sha,
  node: process.version, evidence: 'unchanged upstream TypeScript formatter executed on synthetic fixtures',
  checks_passed: 4,
  short_append: { prior_chars: first.length, common_prefix_chars: lcp(first, appended) },
  long_truncation: { prior_chars: longBefore.length, new_chars: longAfter.length, common_prefix_chars: longPrefix },
  image_only_transcript_chars: imageOnly.length, oversized_tool_result_chars: toolTrim.length,
  limits: ['Character prefixes are not exact token/cache-hit counts',
           'No model, cloud API, real trajectory or runtime privacy evaluation was executed',
           'No claim that long truncation occurs frequently in the published benchmark'],
};
writeFileSync(new URL('./results/ec2_transcript_audit.json', import.meta.url), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
