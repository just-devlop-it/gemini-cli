import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import { ContentGenerator, ContentGeneratorConfig } from './contentGenerator.js';
import { ClaudeAdapter, ClaudeAdapterConfig } from '../adapters/claudeAdapter.js';
import { UserTierId } from '../code_assist/types.js';

export class ClaudeContentGenerator implements ContentGenerator {
  private adapter: ClaudeAdapter;
  public userTier?: UserTierId;

  constructor(config: ContentGeneratorConfig) {
    const region = process.env['CLOUD_ML_REGION'] || 'us-central1';
    const projectId = process.env['ANTHROPIC_VERTEX_PROJECT_ID'];
    
    if (!projectId) {
      throw new Error(
        'ANTHROPIC_VERTEX_PROJECT_ID environment variable is required for Claude models'
      );
    }

    const adapterConfig: ClaudeAdapterConfig = {
      region,
      projectId,
    };

    this.adapter = new ClaudeAdapter(adapterConfig);
    this.userTier = UserTierId.FREE; // Default tier for Claude users
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse> {
    try {
      return await this.adapter.generateContent(request);
    } catch (error) {
      throw new Error(`Claude API error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    try {
      return this.adapter.generateContentStream(request);
    } catch (error) {
      throw new Error(`Claude streaming API error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    try {
      return await this.adapter.countTokens(request);
    } catch (error) {
      throw new Error(`Claude token counting error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    // Claude doesn't support embeddings, delegate to Gemini or throw error
    throw new Error('Embedding not supported for Claude models. Please use Gemini embedding models instead.');
  }
}
