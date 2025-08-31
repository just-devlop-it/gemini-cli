import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import { 
  GenerateContentResponse, 
  GenerateContentParameters,
  CountTokensResponse,
  EmbedContentResponse,
  EmbedContentParameters,
  CountTokensParameters,
  Content,
  Part,
  FunctionCall,
  FinishReason
} from '@google/genai';

export interface ClaudeAdapterConfig {
  region: string;
  projectId: string;
}

export class ClaudeAdapter {
  private client: AnthropicVertex;
  
  constructor(config: ClaudeAdapterConfig) {
    console.log('Debug: Claude adapter config - region:', config.region, 'projectId:', config.projectId);
    this.client = new AnthropicVertex({
      region: config.region,
      projectId: config.projectId,
    });
  }

  async generateContent(request: GenerateContentParameters): Promise<GenerateContentResponse> {
    const claudeRequest = this.convertToClaudeFormat(request);
    const claudeResponse = await this.client.messages.create({
      ...claudeRequest,
      model: request.model,
    });
    return this.convertToGeminiFormat(claudeResponse);
  }

  async *generateContentStream(request: GenerateContentParameters): AsyncGenerator<GenerateContentResponse> {
    const claudeRequest = this.convertToClaudeFormat(request);
    
    try {
      const stream = await this.client.messages.create({
        ...claudeRequest,
        model: request.model,
        stream: true,
      }) as any;

      for await (const chunk of stream) {
        if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
          yield this.convertStreamChunkToGemini(chunk);
        }
      }
    } catch (error) {
      throw new Error(`Claude streaming error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Claude doesn't have a direct token counting API
    // Approximate based on text length (rough estimation: 1 token ≈ 4 characters)
    const contents = Array.isArray(request.contents) ? request.contents as Content[] : [];
    const text = this.extractTextFromContents(contents);
    const approximateTokens = Math.ceil(text.length / 4);
    
    return {
      totalTokens: approximateTokens,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    // Claude doesn't support embeddings through Vertex AI
    // This would need to be handled by a different service
    throw new Error('Embedding not supported for Claude models. Use Gemini embedding models instead.');
  }

  private convertToClaudeFormat(request: GenerateContentParameters): any {
    const messages = [];
    let systemPrompt = '';

    // Extract system instruction
    if (request.config?.systemInstruction) {
      if (typeof request.config.systemInstruction === 'string') {
        systemPrompt = request.config.systemInstruction;
      } else if (typeof request.config.systemInstruction === 'object' && 'parts' in request.config.systemInstruction) {
        // Extract text from parts
        const parts = request.config.systemInstruction.parts || [];
        for (const part of parts) {
          if ('text' in part && part.text) {
            systemPrompt += part.text + ' ';
          }
        }
        systemPrompt = systemPrompt.trim();
      }
    }

    // Convert contents to Claude messages format
    const contents = Array.isArray(request.contents) ? request.contents as Content[] : [];
    for (const content of contents) {
      if ('role' in content && content.role === 'user') {
        messages.push({
          role: 'user',
          content: this.convertPartsToClaudeContent(content.parts || []),
        });
      } else if ('role' in content && content.role === 'model') {
        messages.push({
          role: 'assistant',
          content: this.convertPartsToClaudeContent(content.parts || []),
        });
      }
    }

    const claudeRequest: any = {
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
    };

    if (systemPrompt) {
      claudeRequest.system = systemPrompt;
    }

    if (request.config?.temperature !== undefined) {
      claudeRequest.temperature = request.config.temperature;
    }

    if (request.config?.topP !== undefined) {
      claudeRequest.top_p = request.config.topP;
    }

    // Convert tools if present
    if (request.config?.tools && request.config.tools.length > 0) {
      claudeRequest.tools = this.convertToolsToClaudeFormat(request.config.tools);
    }

    return claudeRequest;
  }

  private convertPartsToClaudeContent(parts: Part[]): any[] {
    const content = [];
    
    for (const part of parts) {
      if ('text' in part && part.text) {
        content.push({
          type: 'text',
          text: part.text,
        });
      } else if ('functionCall' in part && part.functionCall) {
        content.push({
          type: 'tool_use',
          id: `tool_${Date.now()}`,
          name: part.functionCall.name,
          input: part.functionCall.args || {},
        });
      } else if ('functionResponse' in part && part.functionResponse) {
        content.push({
          type: 'tool_result',
          tool_use_id: `tool_${Date.now()}`,
          content: JSON.stringify(part.functionResponse.response),
        });
      }
    }

    return content;
  }

  private convertToolsToClaudeFormat(tools: any[]): any[] {
    const claudeTools = [];
    
    for (const tool of tools) {
      if (tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          claudeTools.push({
            name: func.name,
            description: func.description,
            input_schema: func.parameters || {},
          });
        }
      }
    }

    return claudeTools;
  }

  private convertToGeminiFormat(claudeResponse: any): GenerateContentResponse {
    const parts: Part[] = [];
    
    if (claudeResponse.content) {
      for (const block of claudeResponse.content) {
        if (block.type === 'text') {
          parts.push({ text: block.text });
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: block.name,
              args: block.input,
            } as FunctionCall,
          });
        }
      }
    }

    const response = {
      candidates: [
        {
          content: {
            parts,
            role: 'model',
          },
          finishReason: this.mapFinishReason(claudeResponse.stop_reason),
          index: 0,
          safetyRatings: [],
        },
      ],
      promptFeedback: {
        safetyRatings: [],
      },
      text: parts.find(p => 'text' in p)?.text || '',
      functionCalls: parts.filter(p => 'functionCall' in p).map(p => p.functionCall!),
      data: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
    } as GenerateContentResponse;

    return response;
  }

  private convertStreamChunkToGemini(chunk: any): GenerateContentResponse {
    const response = {
      candidates: [
        {
          content: {
            parts: [{ text: chunk.delta.text }],
            role: 'model',
          },
          finishReason: FinishReason.STOP,
          index: 0,
          safetyRatings: [],
        },
      ],
      promptFeedback: {
        safetyRatings: [],
      },
      text: chunk.delta.text,
      functionCalls: [],
      data: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
    } as GenerateContentResponse;

    return response;
  }

  private mapFinishReason(claudeStopReason: string | undefined): FinishReason {
    switch (claudeStopReason) {
      case 'end_turn':
        return FinishReason.STOP;
      case 'max_tokens':
        return FinishReason.MAX_TOKENS;
      case 'tool_use':
        return FinishReason.STOP;
      default:
        return FinishReason.STOP;
    }
  }

  private extractTextFromContents(contents: Content[]): string {
    let text = '';
    for (const content of contents) {
      if ('parts' in content && content.parts) {
        for (const part of content.parts) {
          if ('text' in part && part.text) {
            text += part.text + ' ';
          }
        }
      }
    }
    return text.trim();
  }
}
