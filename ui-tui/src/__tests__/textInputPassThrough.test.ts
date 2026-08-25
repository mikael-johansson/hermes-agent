import { describe, expect, it } from 'vitest'

import { shouldPassThroughToGlobalHandler, shouldPreserveCtrlJNewline } from '../components/textInput.js'
import { DEFAULT_VOICE_RECORD_KEY, parseVoiceRecordKey } from '../lib/platform.js'

const key = (overrides: Record<string, unknown> = {}) => ({ ctrl: false, meta: false, ...overrides }) as any

describe('shouldPreserveCtrlJNewline', () => {
  it('preserves Ctrl+J as newline in Ghostty even when tmux masks TERM/TERM_PROGRAM', () => {
    expect(
      shouldPreserveCtrlJNewline({
        GHOSTTY_RESOURCES_DIR: '/usr/share/ghostty',
        TERM: 'tmux-256color',
        TERM_PROGRAM: 'tmux'
      })
    ).toBe(true)
  })

  it('keeps bare local POSIX LF-compatible prompts submitting on Ctrl+J', () => {
    expect(shouldPreserveCtrlJNewline({ TERM: 'xterm-256color' })).toBe(false)
  })
})

describe('shouldPassThroughToGlobalHandler', () => {
  it('passes through the configured voice shortcut while composer is focused', () => {
    expect(shouldPassThroughToGlobalHandler('o', key({ ctrl: true }), parseVoiceRecordKey('ctrl+o'))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('r', key({ meta: true }), parseVoiceRecordKey('alt+r'))).toBe(true)
    expect(shouldPassThroughToGlobalHandler(' ', key({ ctrl: true }), parseVoiceRecordKey('ctrl+space'))).toBe(true)
    expect(
      shouldPassThroughToGlobalHandler('', key({ ctrl: true, return: true }), parseVoiceRecordKey('ctrl+enter'))
    ).toBe(true)
  })

  it('keeps the default pass-through when no custom key is provided', () => {
    // Default is Ctrl+Space now; Ctrl+B/F are reserved for readline
    // cursor movement and no longer belong to the voice pass-through.
    expect(shouldPassThroughToGlobalHandler(' ', key({ ctrl: true }), DEFAULT_VOICE_RECORD_KEY)).toBe(true)
    expect(shouldPassThroughToGlobalHandler(' ', key({ ctrl: true }))).toBe(true)
    // Ctrl+B is NOT a voice shortcut anymore — it stays in the composer.
    expect(shouldPassThroughToGlobalHandler('b', key({ ctrl: true }), DEFAULT_VOICE_RECORD_KEY)).toBe(false)
    expect(shouldPassThroughToGlobalHandler('f', key({ ctrl: true }))).toBe(false)
  })

  it('does not swallow ordinary typing keys', () => {
    expect(shouldPassThroughToGlobalHandler('h', key(), parseVoiceRecordKey('ctrl+o'))).toBe(false)
    expect(shouldPassThroughToGlobalHandler('o', key(), parseVoiceRecordKey('ctrl+o'))).toBe(false)
  })

  it('always passes through non-voice global control keys', () => {
    expect(shouldPassThroughToGlobalHandler('c', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('x', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('o', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ escape: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ tab: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageUp: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageDown: true }))).toBe(true)
  })
})
