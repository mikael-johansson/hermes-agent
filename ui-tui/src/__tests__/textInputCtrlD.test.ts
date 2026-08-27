import { describe, expect, it } from 'vitest'

import { ctrlDEffect } from '../components/textInput.js'

// Ctrl+D is readline's kill-char: delete the next grapheme cluster (or an
// active selection) at the cursor. On an EMPTY line it signals EOF, which
// the global input handler turns into the traditional idle exit / dashboard
// new-session. The global handler independently refuses to idle-exit while
// a draft is present, so a typed line can never kill the session.

describe('ctrlDEffect — kill-char', () => {
  it('deletes the character at the cursor, keeping the rest', () => {
    expect(ctrlDEffect('hello', 1)).toEqual({ exit: false, newCursor: 1, newValue: 'hllo' })
  })

  it('deletes from mid-line, leaving the trailing text intact', () => {
    expect(ctrlDEffect('hello world', 6)).toEqual({ exit: false, newCursor: 6, newValue: 'hello orld' })
  })

  it('is a no-op when the cursor is already at the end of a non-empty line', () => {
    expect(ctrlDEffect('hello', 5)).toEqual({ exit: false, newCursor: 5, newValue: 'hello' })
  })

  it('reports EOF (exit) for an empty line', () => {
    expect(ctrlDEffect('', 0)).toEqual({ exit: true, newCursor: 0, newValue: '' })
  })

  it('keeps combining sequences together (no half-grapheme deletes)', () => {
    // e + U+0301 (combining acute) = é, a single grapheme cluster.
    const e = 'e\u0301'
    expect(ctrlDEffect(e + 'x', 0)).toEqual({ exit: false, newCursor: 0, newValue: 'x' })
  })

  it('handles an emoji (multi-code-unit grapheme) without tearing it', () => {
    // 👍 is two UTF-16 code units; a naive slice(0,c)+slice(c+1) would tear it.
    expect(ctrlDEffect('\u{1F44D}abc', 0)).toEqual({ exit: false, newCursor: 0, newValue: 'abc' })
  })

  it('deletes a whole active selection and parks the cursor at its start', () => {
    expect(ctrlDEffect('hello world', 6, { start: 0, end: 6 })).toEqual({
      exit: false,
      newCursor: 0,
      newValue: 'world'
    })
  })

  it('ignores a degenerate (zero-width) selection and falls back to kill-char', () => {
    expect(ctrlDEffect('hello', 1, { start: 1, end: 1 })).toEqual({ exit: false, newCursor: 1, newValue: 'hllo' })
  })

  it('does not throw on a reversed (un-normalized) range — caller normalizes first', () => {
    // selRange() always normalizes start <= end before calling this; this
    // pins only that a raw reversed range degrades gracefully (no throw, no
    // negative-slice crash) rather than pinning the degenerate result shape.
    expect(() => ctrlDEffect('hello world', 0, { start: 11, end: 6 })).not.toThrow()
  })
})
