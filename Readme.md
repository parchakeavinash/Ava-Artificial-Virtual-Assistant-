
                    ┌──────────────────────┐
                    │      Voice Input     │
                    └──────────┬───────────┘
                               ↓
                         sarvamAI
                               ↓
                          Transcript
                               ↓
                    ┌──────────────────────┐
                    │      LangGraph       │
                    │   Agent Orchestrator │
                    └──────────┬───────────┘
                               ↓
                          ┌─────────┐
                          │ gemini   │
                          └────┬────┘
                               ↓
                       Tool decision
                               ↓
                       ┌─────────────┐
                       │ MCP Client  │
                       └──────┬──────┘
                              ↓
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
       Web MCP            DB MCP           Memory MCP
             ↓                ↓                ↓
        Firecrawl         Supabase          pgvector
             │                │                │
             └────────────────┼────────────────┘
                              ↓
                         Tool Results
                              ↓
                           gemini
                              ↓
                        Final Response
                              ↓
                            TTS
                              ↓
                       Streaming Audio
                              ↓
                            User
## phase 1:
output:
<img width="966" height="665" alt="image" src="https://github.com/user-attachments/assets/c2331103-0365-4337-ab4f-bff7dea1a917" />
## Phase 2:
output:
<img width="977" height="666" alt="image" src="https://github.com/user-attachments/assets/5f3312b9-68a1-4b6e-83be-6354fd6c51f6" />

