"""
Agent: Interactive AI
Purpose: General interactive AI assistance for tasks, clarifications, and multi-step workflows
"""

from typing import Dict, Any

# System prompt for the Interactive AI agent
INTERACTIVE_AI_SYSTEM_PROMPT = """
You are an expert Interactive AI Assistant specializing in legal workflow support and general task assistance.
Your role is to help users with a wide range of tasks related to legal document work and general assistance.

## Core Capabilities

1. **General Chat**
   - Answer general questions
   - Provide information and guidance
   - Engage in professional conversation
   - Clarify concepts and processes

2. **Task Assistance**
   - Help structure tasks
   - Provide step-by-step guidance
   - Suggest approaches and methodologies
   - Assist with planning and organization

3. **Clarification Requests**
   - Explain unclear responses
   - Provide additional context
   - Rephrase information
   - Answer follow-up questions

4. **Multi-Step Workflows**
   - Guide through complex processes
   - Coordinate multiple tasks
   - Manage sequential steps
   - Track progress and status

## Interaction Guidelines

### Helpfulness
- Be genuinely helpful and supportive
- Anticipate user needs
- Offer relevant suggestions
- Go beyond minimal responses

### Clarity
- Communicate clearly and concisely
- Structure information logically
- Use appropriate formatting
- Confirm understanding

### Professionalism
- Maintain professional tone
- Respect user expertise
- Be courteous and patient
- Acknowledge limitations

### Adaptability
- Adjust to user needs
- Match user's communication style
- Scale detail appropriately
- Handle ambiguity gracefully

## Response Format

Structure responses based on the task:

1. **Acknowledgment**: Confirm understanding of request
2. **Main Response**: Direct answer or guidance
3. **Additional Help**: Related suggestions
4. **Next Steps**: Recommended actions

## Constraints

- Do not provide legal advice
- Do not make decisions for users
- Stay within professional boundaries
- Acknowledge when specialist help is needed
- Maintain confidentiality
"""

# Prompt templates for different sub-intents
GENERAL_CHAT_PROMPT = """
Engage in a helpful conversation responding to the user's message.

User Message: {user_message}

Conversation History:
{conversation_history}

Context:
{context}

Respond:

1. **Direct Response**
   - Address the user's message directly
   - Provide helpful information
   - Answer questions clearly
   - Engage professionally

2. **Contextual Awareness**
   - Reference previous conversation
   - Build on established context
   - Maintain continuity
   - Acknowledge user expertise

3. **Helpful Additions**
   - Related information
   - Useful suggestions
   - Relevant resources
   - Best practices

4. **Engagement**
   - Ask clarifying questions if needed
   - Invite further discussion
   - Offer additional assistance
   - Maintain conversation flow

Keep responses professional, helpful, and appropriately detailed.
"""

TASK_ASSISTANCE_PROMPT = """
Provide assistance with the following task.

Task Description: {task_description}
Task Type: {task_type}
Current Stage: {current_stage}
Available Resources: {available_resources}

Context:
{context}

Provide:

1. **Task Understanding**
   - Clarify task objectives
   - Confirm scope and boundaries
   - Identify key requirements
   - Note constraints

2. **Approach Guidance**
   - Suggested methodology
   - Step-by-step breakdown
   - Alternative approaches
   - Best practices

3. **Step-by-Step Instructions**
   - Detailed steps
   - Prerequisites for each step
   - Expected outcomes
   - Checkpoints

4. **Resources and Tools**
   - Required resources
   - Helpful tools
   - Reference materials
   - Templates if applicable

5. **Tips and Recommendations**
   - Common pitfalls to avoid
   - Efficiency tips
   - Quality considerations
   - Expert suggestions

6. **Timeline Guidance**
   - Estimated time needed
   - Milestone suggestions
   - Critical path items
   - Buffer considerations

7. **Quality Checks**
   - Validation steps
   - Review checkpoints
   - Success criteria
   - Completion indicators

8. **Next Steps**
   - Immediate actions
   - Follow-up tasks
   - Escalation triggers
   - Support resources
"""

CLARIFICATION_REQUEST_PROMPT = """
Provide clarification on the requested topic or previous response.

Clarification Request: {clarification_request}
Original Topic/Response: {original_content}
Clarification Type: {clarification_type}

Context:
{context}

Provide:

1. **Clarification Focus**
   - Identify what needs clarification
   - Understand user's perspective
   - Address specific confusion
   - Confirm understanding of request

2. **Clear Explanation**
   - Rephrase in simpler terms
   - Break down complex concepts
   - Use examples or analogies
   - Provide step-by-step breakdown

3. **Additional Context**
   - Background information
   - Related concepts
   - Why it matters
   - How it fits in broader picture

4. **Multiple Perspectives**
   - Different ways to understand
   - Alternative explanations
   - Visual descriptions if helpful
   - Practical applications

5. **Confirmation**
   - Verify clarification is helpful
   - Ask if more detail needed
   - Offer related clarifications
   - Ensure understanding

6. **Related Information**
   - Connected topics
   - Prerequisites to understand
   - Follow-up concepts
   - Resources for deeper learning
"""

MULTI_STEP_WORKFLOW_PROMPT = """
Guide the user through a multi-step workflow.

Workflow Type: {workflow_type}
Current Step: {current_step}
Total Steps: {total_steps}
Workflow Status: {workflow_status}

Context:
{context}

Previous Steps:
{previous_steps}

Provide:

1. **Workflow Overview**
   - Purpose of workflow
   - Expected outcome
   - Total steps and timeline
   - Success criteria

2. **Current Step Guidance**
   - Step objective
   - Detailed instructions
   - Required inputs
   - Expected outputs

3. **Step Instructions**
   - Action items
   - Decision points
   - Validation checks
   - Common issues

4. **Progress Tracking**
   - Steps completed
   - Current position
   - Remaining steps
   - Milestone status

5. **Navigation Support**
   - How to proceed
   - How to go back if needed
   - How to skip (if applicable)
   - How to restart

6. **Troubleshooting**
   - Common problems
   - Solutions
   - When to seek help
   - Escalation paths

7. **Next Step Preview**
   - What comes next
   - Preparation needed
   - Dependencies
   - Expected duration

8. **Workflow Management**
   - Save/restore options
   - Status updates
   - Change handling
   - Completion criteria

9. **Support Resources**
   - Help documentation
   - Examples
   - Templates
   - Contact information
"""


def get_prompt_for_sub_intent(sub_intent: str, parameters: Dict[str, Any]) -> str:
    """
    Get the appropriate prompt template for the sub-intent.
    
    Args:
        sub_intent: The specific task type
        parameters: Parameters for the prompt
    
    Returns:
        Formatted prompt string
    """
    if sub_intent == "general_chat":
        return GENERAL_CHAT_PROMPT.format(
            user_message=parameters.get("user_message", ""),
            conversation_history=parameters.get("conversation_history", "No previous conversation."),
            context=parameters.get("context", "General assistance.")
        )
    
    elif sub_intent == "task_assistance":
        return TASK_ASSISTANCE_PROMPT.format(
            task_description=parameters.get("task_description", ""),
            task_type=parameters.get("task_type", "general"),
            current_stage=parameters.get("current_stage", "planning"),
            available_resources=parameters.get("available_resources", "standard resources"),
            context=parameters.get("context", "Task assistance.")
        )
    
    elif sub_intent == "clarification_request":
        return CLARIFICATION_REQUEST_PROMPT.format(
            clarification_request=parameters.get("clarification_request", ""),
            original_content=parameters.get("original_content", ""),
            clarification_type=parameters.get("clarification_type", "general"),
            context=parameters.get("context", "Clarification needed.")
        )
    
    elif sub_intent == "multi_step_workflow":
        return MULTI_STEP_WORKFLOW_PROMPT.format(
            workflow_type=parameters.get("workflow_type", "general"),
            current_step=parameters.get("current_step", 1),
            total_steps=parameters.get("total_steps", 1),
            workflow_status=parameters.get("workflow_status", "in_progress"),
            context=parameters.get("context", "Workflow guidance."),
            previous_steps=parameters.get("previous_steps", "None")
        )
    
    else:
        # Default to general chat
        return GENERAL_CHAT_PROMPT.format(
            user_message=parameters.get("user_message", "How can I help you today?"),
            conversation_history=parameters.get("conversation_history", "No previous conversation."),
            context=parameters.get("context", "General assistance.")
        )
