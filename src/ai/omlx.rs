// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AI provider for oMLX — a local LLM server on macOS that exposes an
//! OpenAI-compatible chat-completions endpoint.
//!
//! Default endpoint: `http://localhost:1993/v1/chat/completions`
//!
//! Configuration example (Settings.toml):
//! ```toml
//! [ai]
//! provider = "omlx"
//! model = "gemma-4-27b-it"
//!
//! [ai.omlx]
//! base_url = "http://localhost:1993/v1/chat/completions"
//! context_window_size = 131072
//! max_tokens = 8192
//! ```

use crate::ai::token_budget::TokenBudget;
use crate::ai::{
    AiProvider, AiRequest, AiResponse, AiResponseFormat, AiRole, AiUsage, ProviderCapabilities,
    ToolCall,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

// ── Wire types (OpenAI-compatible subset) ────────────────────────────

#[derive(Debug, Serialize)]
struct OmlxRequest {
    model: String,
    messages: Vec<OmlxMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OmlxTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OmlxMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OmlxToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OmlxToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OmlxToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OmlxToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Clone)]
struct OmlxTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OmlxFunction,
}

#[derive(Debug, Serialize, Clone)]
struct OmlxFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct OmlxResponse {
    choices: Vec<OmlxChoice>,
    #[serde(default)]
    usage: Option<OmlxUsage>,
}

#[derive(Debug, Deserialize)]
struct OmlxChoice {
    message: OmlxMessage,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OmlxUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

// ── Provider implementation ──────────────────────────────────────────

/// Default oMLX endpoint (localhost, port 1993).
const DEFAULT_BASE_URL: &str = "http://localhost:1993/v1/chat/completions";

pub struct OmlxClient {
    api_key: String,
    model: String,
    base_url: String,
    context_window_size: usize,
    max_tokens: u32,
    client: Client,
}

impl OmlxClient {
    pub fn new(
        base_url: String,
        model: String,
        context_window_size: usize,
        max_tokens: u32,
        api_timeout_secs: u64,
    ) -> Self {
        let api_key = std::env::var("OMLX_API_KEY")
            .or_else(|_| std::env::var("LLM_API_KEY"))
            .unwrap_or_default();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(api_timeout_secs))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            api_key,
            model,
            base_url,
            context_window_size,
            max_tokens,
            client,
        }
    }

    pub fn default_base_url() -> String {
        DEFAULT_BASE_URL.to_string()
    }

    /// Returns a sensible context-window size based on the model name.
    pub fn default_context_window_for_model(model: &str) -> usize {
        let m = model.to_lowercase();
        if m.contains("qwen") {
            // Qwen 3 / 3.5 models default to 128K context
            131_072
        } else if m.contains("gemma") {
            // Gemma 4 27B — 128K context
            131_072
        } else if m.contains("llama") {
            // Llama 3.x — 128K context
            131_072
        } else if m.contains("mistral") {
            // Mistral / Mixtral — 32K–128K depending on variant
            131_072
        } else {
            // Safe fallback
            131_072
        }
    }

    /// Returns a sensible max-output-tokens default based on the model name.
    pub fn default_max_tokens_for_model(model: &str) -> u32 {
        let m = model.to_lowercase();
        if m.contains("qwen") {
            // Qwen 3.5 supports longer outputs
            16_384
        } else if m.contains("gemma") {
            8192
        } else {
            8192
        }
    }

    async fn post_request(&self, body: &Value) -> Result<OmlxResponse> {
        let mut req = self.client.post(&self.base_url).json(body);

        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let res = req.send().await.map_err(|e| {
                if e.is_connect() {
                    anyhow::anyhow!(
                        "Could not connect to oMLX at {}. Is the server running?",
                        self.base_url
                    )
                } else if e.is_timeout() {
                    anyhow::anyhow!(
                        "oMLX request timed out ({}). The model may need more time — \
                         consider increasing api_timeout_secs.",
                        self.base_url
                    )
                } else {
                    anyhow::anyhow!("oMLX request failed: {e}")
                }
            })?;

        if !res.status().is_success() {
            let status = res.status();
            let body_text = res.text().await.unwrap_or_default();
            anyhow::bail!("oMLX returned HTTP {status}: {body_text}");
        }

        let body_text = res
            .text()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read oMLX response body: {e}"))?;

        serde_json::from_str::<OmlxResponse>(&body_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse oMLX response: {e}\nBody: {body_text}"))
    }
}

// ── Translation helpers ──────────────────────────────────────────────

fn translate_request(request: AiRequest, model: &str, max_tokens: u32) -> OmlxRequest {
    let mut messages = Vec::new();

    // Extract schema before consuming response_format
    let request_schema = request
        .response_format
        .as_ref()
        .and_then(|rf| match rf {
            AiResponseFormat::Json { schema } => schema.clone(),
            _ => None,
        });

    if let Some(system_text) = request.system {
        messages.push(OmlxMessage {
            role: "system".to_string(),
            content: Some(system_text),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for msg in request.messages {
        match msg.role {
            AiRole::System => messages.push(OmlxMessage {
                role: "system".to_string(),
                content: msg.content,
                tool_calls: None,
                tool_call_id: None,
            }),
            AiRole::User => messages.push(OmlxMessage {
                role: "user".to_string(),
                content: msg.content,
                tool_calls: None,
                tool_call_id: None,
            }),
            AiRole::Assistant => messages.push(OmlxMessage {
                role: "assistant".to_string(),
                content: msg.content,
                tool_calls: msg.tool_calls.map(|tc| {
                    tc.into_iter()
                        .map(|t| OmlxToolCall {
                            id: t.id,
                            tool_type: "function".to_string(),
                            function: OmlxToolCallFunction {
                                name: t.function_name,
                                arguments: serde_json::to_string(&t.arguments)
                                    .unwrap_or_default(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: None,
            }),
            AiRole::Tool => messages.push(OmlxMessage {
                role: "tool".to_string(),
                content: msg.content,
                tool_calls: None,
                tool_call_id: msg.tool_call_id,
            }),
        }
    }

    let tools = request.tools.and_then(|t| {
        if t.is_empty() {
            None
        } else {
            Some(
                t.into_iter()
                    .map(|tool| OmlxTool {
                        tool_type: "function".to_string(),
                        function: OmlxFunction {
                            name: tool.name,
                            description: tool.description,
                            parameters: tool.parameters,
                        },
                    })
                    .collect(),
            )
        }
    });

    let response_format = request.response_format.map(|rf| match rf {
        AiResponseFormat::Json { .. } => serde_json::json!({"type": "json_object"}),
        AiResponseFormat::Text => serde_json::json!({"type": "text"}),
    });

    // OpenAI-compatible APIs (and local models especially) require "json" to
    // appear in at least one message when using response_format: json_object.
    // Additionally, local models benefit from seeing the expected JSON schema
    // explicitly in the prompt so they know what structure to produce.
    if response_format
        .as_ref()
        .is_some_and(|rf| rf["type"] == "json_object")
    {
        // Build a schema hint if one was provided in the original request
        let schema_hint = request_schema.map(|s| {
            format!(
                "\nYou MUST respond with valid JSON matching this schema:\n```json\n{}\n```",
                serde_json::to_string_pretty(&s).unwrap_or_else(|_| s.to_string())
            )
        });

        let has_json = messages.iter().any(|m| {
            m.content
                .as_ref()
                .is_some_and(|c| c.to_lowercase().contains("json"))
        });

        if !has_json || schema_hint.is_some() {
            if let Some(system_msg) = messages.iter_mut().find(|m| m.role == "system") {
                let content = system_msg.content.get_or_insert_default();
                if let Some(hint) = &schema_hint {
                    content.push_str(hint);
                } else {
                    content.push_str("\nRespond in JSON format.");
                }
            } else {
                let text = schema_hint.unwrap_or_else(|| "Respond in JSON format.".to_string());
                messages.insert(
                    0,
                    OmlxMessage {
                        role: "system".to_string(),
                        content: Some(text),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                );
            }
        }
    }

    OmlxRequest {
        model: model.to_string(),
        messages,
        tools,
        temperature: request.temperature,
        max_tokens: Some(max_tokens),
        response_format,
    }
}

fn translate_response(resp: OmlxResponse) -> Result<AiResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("oMLX returned no choices"))?;

    let content = choice.message.content;
    let tool_calls = choice.message.tool_calls.map(|tc| {
        tc.into_iter()
            .map(|t| {
                let arguments: Value =
                    serde_json::from_str(&t.function.arguments).unwrap_or(Value::Null);
                ToolCall {
                    id: t.id,
                    function_name: t.function.name,
                    arguments,
                    thought_signature: None,
                }
            })
            .collect()
    });

    let usage = resp.usage.map(|u| AiUsage {
        prompt_tokens: u.prompt_tokens as usize,
        completion_tokens: u.completion_tokens as usize,
        total_tokens: u.total_tokens as usize,
        cached_tokens: None,
    });

    Ok(AiResponse {
        content,
        thought: None,
        tool_calls,
        usage,
    })
}

fn estimate_tokens(request: &AiRequest) -> usize {
    let mut total = 0;
    if let Some(system) = &request.system {
        total += TokenBudget::estimate_tokens(system);
    }
    for msg in &request.messages {
        if let Some(content) = &msg.content {
            total += TokenBudget::estimate_tokens(content);
        }
        if let Some(tool_calls) = &msg.tool_calls {
            for call in tool_calls {
                total += TokenBudget::estimate_tokens(&call.function_name);
                total += TokenBudget::estimate_tokens(&call.arguments.to_string());
            }
        }
    }
    if let Some(tools) = &request.tools {
        for tool in tools {
            total += TokenBudget::estimate_tokens(&tool.name);
            total += TokenBudget::estimate_tokens(&tool.description);
            total += TokenBudget::estimate_tokens(&tool.parameters.to_string());
        }
    }
    total
}

// ── AiProvider trait ─────────────────────────────────────────────────

#[async_trait]
impl AiProvider for OmlxClient {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let tag = request
            .context_tag
            .as_deref()
            .unwrap_or("")
            .to_string();
        tracing::info!("{tag} Sending oMLX request to {} (model={})", self.base_url, self.model);

        let omlx_req = translate_request(request, &self.model, self.max_tokens);
        let body = serde_json::to_value(&omlx_req)?;
        let resp = self.post_request(&body).await?;

        if let Some(u) = &resp.usage {
            tracing::info!(
                "{tag} oMLX response received. Tokens: in={}, out={}",
                u.prompt_tokens,
                u.completion_tokens
            );
        } else {
            tracing::info!("{tag} oMLX response received (no usage reported).");
        }

        translate_response(resp)
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        estimate_tokens(request)
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: self.context_window_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiMessage, AiTool};
    use serde_json::json;

    #[test]
    fn test_translate_request_basic() {
        let req = AiRequest {
            system: Some("You are helpful.".to_string()),
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("Hello!".to_string()),
                thought: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: Some(0.7),
            response_format: None,
            context_tag: None,
        };

        let omlx_req = translate_request(req, "gemma-4-27b-it", 8192);
        assert_eq!(omlx_req.model, "gemma-4-27b-it");
        assert_eq!(omlx_req.messages.len(), 2);
        assert_eq!(omlx_req.messages[0].role, "system");
        assert_eq!(omlx_req.messages[1].role, "user");
        assert_eq!(omlx_req.temperature, Some(0.7));
        assert_eq!(omlx_req.max_tokens, Some(8192));
    }

    #[test]
    fn test_translate_request_with_tools() {
        let req = AiRequest {
            system: None,
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("Use a tool.".to_string()),
                thought: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Some(vec![AiTool {
                name: "get_weather".to_string(),
                description: "Get the weather".to_string(),
                parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }]),
            temperature: None,
            response_format: None,
            context_tag: None,
        };

        let omlx_req = translate_request(req, "gemma-4-27b-it", 4096);
        assert!(omlx_req.tools.is_some());
        let tools = omlx_req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_translate_response_basic() {
        let raw = OmlxResponse {
            choices: vec![OmlxChoice {
                message: OmlxMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello there!".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OmlxUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };

        let resp = translate_response(raw).unwrap();
        assert_eq!(resp.content, Some("Hello there!".to_string()));
        assert!(resp.tool_calls.is_none());
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_translate_response_no_usage() {
        let raw = OmlxResponse {
            choices: vec![OmlxChoice {
                message: OmlxMessage {
                    role: "assistant".to_string(),
                    content: Some("Hi".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };

        let resp = translate_response(raw).unwrap();
        assert_eq!(resp.content, Some("Hi".to_string()));
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_translate_response_with_tool_calls() {
        let raw = OmlxResponse {
            choices: vec![OmlxChoice {
                message: OmlxMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![OmlxToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: OmlxToolCallFunction {
                            name: "get_weather".to_string(),
                            arguments: r#"{"city":"Tokyo"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(OmlxUsage {
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
            }),
        };

        let resp = translate_response(raw).unwrap();
        assert!(resp.content.is_none());
        let calls = resp.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "get_weather");
        assert_eq!(calls[0].arguments, json!({"city": "Tokyo"}));
    }

    #[test]
    fn test_translate_request_json_schema_injected() {
        let schema = json!({
            "type": "object",
            "properties": {
                "selected_prompts": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["selected_prompts"]
        });

        let req = AiRequest {
            system: Some("You are a selector.".to_string()),
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("Pick items.".to_string()),
                thought: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: Some(0.0),
            response_format: Some(AiResponseFormat::Json {
                schema: Some(schema),
            }),
            context_tag: None,
        };

        let omlx_req = translate_request(req, "gemma-4-27b-it", 8192);
        // System message should have the schema appended
        let system_content = omlx_req.messages[0].content.as_ref().unwrap();
        assert!(
            system_content.contains("selected_prompts"),
            "Schema should be injected into system prompt"
        );
        assert!(
            system_content.contains("You MUST respond with valid JSON"),
            "JSON instruction should be injected"
        );
        // response_format should still be set
        assert!(omlx_req.response_format.is_some());
    }

    #[test]
    fn test_translate_request_json_no_schema_injects_fallback() {
        let req = AiRequest {
            system: Some("You are helpful.".to_string()),
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("Give me data.".to_string()),
                thought: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: None,
            response_format: Some(AiResponseFormat::Json { schema: None }),
            context_tag: None,
        };

        let omlx_req = translate_request(req, "qwen3.5-27b", 4096);
        let system_content = omlx_req.messages[0].content.as_ref().unwrap();
        assert!(
            system_content.contains("JSON"),
            "Should inject JSON instruction when no schema provided"
        );
    }
}
