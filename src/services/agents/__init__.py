from src.services.orchestrator.models import AgentType
from src.services.agents.doc_info_agent import DocumentInformationAgent
from src.services.agents.risk_agent import RiskComplianceAgent
from src.services.agents.doc_chat_agent import DocChatAgent

# Registry of all available agents
AGENT_REGISTRY = {
    AgentType.DOCUMENT_INFORMATION: DocumentInformationAgent,
    AgentType.RISK_COMPLIANCE: RiskComplianceAgent,
    AgentType.DOC_CHAT: DocChatAgent,
    # Add other agents here as they are implemented
}
