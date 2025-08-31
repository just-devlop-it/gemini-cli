/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType, Config } from '@google/gemini-cli-core';
import { USER_SETTINGS_PATH } from './config/settings.js';
import { validateAuthMethod } from './config/auth.js';

function getAuthTypeFromEnv(): AuthType | undefined {
  if (process.env['GOOGLE_GENAI_USE_GCA'] === 'true') {
    return AuthType.LOGIN_WITH_GOOGLE;
  }
  if (process.env['GOOGLE_GENAI_USE_VERTEXAI'] === 'true') {
    return AuthType.USE_VERTEX_AI;
  }
  if (process.env['ANTHROPIC_VERTEX_PROJECT_ID']) {
    return AuthType.USE_CLAUDE;
  }
  if (process.env['GEMINI_API_KEY']) {
    return AuthType.USE_GEMINI;
  }
  return undefined;
}

export async function validateNonInteractiveAuth(
  configuredAuthType: AuthType | undefined,
  useExternalAuth: boolean | undefined,
  nonInteractiveConfig: Config,
) {
  // Check if using Claude model and force USE_CLAUDE auth
  const currentModel = nonInteractiveConfig.getModel();
  const isClaudeModel = currentModel.includes('claude');
  
  let effectiveAuthType = configuredAuthType || getAuthTypeFromEnv();
  
  // Force USE_CLAUDE for Claude models
  if (isClaudeModel && process.env['ANTHROPIC_VERTEX_PROJECT_ID']) {
    effectiveAuthType = AuthType.USE_CLAUDE;
  }

  // Debug logging
  console.log('Debug: configuredAuthType =', configuredAuthType);
  console.log('Debug: getAuthTypeFromEnv() =', getAuthTypeFromEnv());
  console.log('Debug: currentModel =', currentModel);
  console.log('Debug: isClaudeModel =', isClaudeModel);
  console.log('Debug: effectiveAuthType =', effectiveAuthType);
  console.log('Debug: ANTHROPIC_VERTEX_PROJECT_ID =', process.env['ANTHROPIC_VERTEX_PROJECT_ID']);

  if (!effectiveAuthType) {
    console.error(
      `Please set an Auth method in your ${USER_SETTINGS_PATH} or specify one of the following environment variables before running: GEMINI_API_KEY, GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_GENAI_USE_GCA, ANTHROPIC_VERTEX_PROJECT_ID`,
    );
    process.exit(1);
  }

  if (!useExternalAuth) {
    const err = validateAuthMethod(effectiveAuthType);
    if (err != null) {
      console.error(err);
      process.exit(1);
    }
  }

  await nonInteractiveConfig.refreshAuth(effectiveAuthType);
  return nonInteractiveConfig;
}
