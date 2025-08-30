/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export const DEFAULT_GEMINI_MODEL = 'gemini-2.5-pro';
export const DEFAULT_GEMINI_FLASH_MODEL = 'gemini-2.5-flash';
export const DEFAULT_GEMINI_FLASH_LITE_MODEL = 'gemini-2.5-flash-lite';

export const DEFAULT_GEMINI_EMBEDDING_MODEL = 'gemini-embedding-001';

// Claude models - Latest available models on Vertex AI
export const CLAUDE_MODELS = {
  // Claude 4 series - Latest flagship models
  'claude-opus-4-1': 'claude-opus-4-1',
  'claude-opus-4': 'claude-opus-4', 
  'claude-sonnet-4': 'claude-sonnet-4',
  
  // Claude 3.7 series - Extended thinking capabilities
  'claude-3-7-sonnet': 'claude-3-7-sonnet',
  
  // Claude 3.5 series - Current generation
  'claude-3-5-sonnet-v2': 'claude-3-5-sonnet-v2',
  'claude-3-5-sonnet': 'claude-3-5-sonnet',
  'claude-3-5-haiku': 'claude-3-5-haiku',
  
  // Claude 3 series - Previous generation (legacy)
  'claude-3-opus': 'claude-3-opus@20240229',
  'claude-3-sonnet': 'claude-3-sonnet@20240229', 
  'claude-3-haiku': 'claude-3-haiku@20240307',
} as const;

export const DEFAULT_CLAUDE_MODEL = CLAUDE_MODELS['claude-3-7-sonnet'];
