"""
Intent-Based Chatbot Agent using LangGraph and Pydantic
Handles user queries by classifying intent and routing to appropriate tools
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Literal, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from services.token_usage_service import TokenUsageService

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=openai_api_key
)

logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

# -----------------------------------------------------------------------------
# SOCIAL MEDIA
# -----------------------------------------------------------------------------

class SocialMediaPayload(BaseModel):
    platform: Optional[List[Literal[
        "facebook",
        "instagram",
        "youtube",
        "linkedin",
        "twitter",
        "pinterest"
    ]]] = None

    content_type: Optional[Literal["post", "reel", "video", "story", "carousel"]] = None
    idea: Optional[str] = None

    media: Optional[Literal["upload", "generate"]] = None
    media_file: Optional[str] = None

    date: Optional[datetime] = None
    task: Optional[Literal["draft", "schedule", "edit", "delete"]] = None
    content: Optional[str] = None  # Content ID from created_content table after generation

# -----------------------------------------------------------------------------
# BLOG
# -----------------------------------------------------------------------------

class BlogPayload(BaseModel):
    platform: Optional[Literal["wordpress", "shopify", "wix", "html"]] = None
    topic: Optional[str] = None
    length: Optional[Literal["short", "medium", "long"]] = None

    media: Optional[Literal["generate", "upload"]] = None
    media_file: Optional[str] = None

    date: Optional[datetime] = None
    task: Optional[Literal["draft", "schedule", "save"]] = None

# -----------------------------------------------------------------------------
# EMAIL
# -----------------------------------------------------------------------------

class EmailPayload(BaseModel):
    email_address: Optional[EmailStr] = None
    content: Optional[str] = None

    attachments: Optional[List[str]] = None

    task: Optional[Literal["send", "save", "schedule"]] = None
    date: Optional[datetime] = None

# -----------------------------------------------------------------------------
# WHATSAPP MESSAGE
# -----------------------------------------------------------------------------

class WhatsAppPayload(BaseModel):
    phone_number: Optional[str] = Field(
        default=None,
        description="Phone number with country code, e.g. +919876543210"
    )

    text: Optional[str] = None
    attachment: Optional[str] = None

    task: Optional[Literal["send", "schedule", "save"]] = None
    date: Optional[datetime] = None

# -----------------------------------------------------------------------------
# ADS
# -----------------------------------------------------------------------------

class AdsPayload(BaseModel):
    platform: Optional[Literal["meta", "google", "linkedin", "youtube"]] = None
    objective: Optional[str] = None
    audience: Optional[str] = None
    budget: Optional[str] = None
    creative: Optional[str] = None

    date: Optional[datetime] = None
    task: Optional[Literal["draft", "schedule", "launch"]] = None

# -----------------------------------------------------------------------------
# CONTENT GENERATION
# -----------------------------------------------------------------------------

class ContentGenerationPayload(BaseModel):
    type: Literal["social_media", "blog", "email", "whatsapp", "ads"]

    social_media: Optional[SocialMediaPayload] = None
    blog: Optional[BlogPayload] = None
    email: Optional[EmailPayload] = None
    whatsapp: Optional[WhatsAppPayload] = None
    ads: Optional[AdsPayload] = None

# =============================================================================
# ANALYTICS
# =============================================================================

class AnalyticsPayload(BaseModel):
    query: Optional[str] = None
    platform: Optional[str] = None
    date_range: Optional[str] = None

# =============================================================================
# LEADS MANAGEMENT
# =============================================================================

class addleadPayload(BaseModel):
    lead_name: Optional[str] = None
    lead_email: Optional[EmailStr] = None
    lead_phone: Optional[str] = None
    platform: Optional[str] = None
    status: Optional[str] = None
    lead_id: Optional[str] = None
    remarks: Optional[str] = None
    follow_up: Optional[datetime] = None

class updateleadPayload(BaseModel):
    lead_name: Optional[str] = None
    lead_email: Optional[EmailStr] = None
    lead_phone: Optional[str] = None
    platform: Optional[str] = None
    remarks: Optional[str] = None
    follow_up: Optional[datetime] = None

class searchleadPayload(BaseModel):
    key: Optional[str] = None
    date_range: Optional[str] = None
    lead_id: Optional[str] = None 

class deleteleadPayload(BaseModel):
    lead_id: Optional[str] = None
    lead_name: Optional[str] = None
    confirm_delete: Optional[bool] = None

class LeadsInsightsPayload(BaseModel):
    lead_id: Optional[str] = None
    lead_name: Optional[str] = None
    platform: Optional[str] = None
    date_range: Optional[str] = Field(
        default=None,
        description="Example: today, yesterday, last 7 days, this week, last month"
    )
    insight_type: Optional[str] = Field(
        default=None,
        description="Type of insight: summary, conversion, engagement, performance, etc."
    )


class LeadsManagementPayload(BaseModel):
    action: Optional[
        Literal[
            "add_lead",
            "update_lead",
            "search_lead",
            "delete_lead",
            "insight"
        ]
    ] = None

    insight: Optional[LeadsInsightsPayload] = None
    add_lead: Optional[addleadPayload] = None
    update_lead: Optional[updateleadPayload] = None
    search_lead: Optional[searchleadPayload] = None
    delete_lead: Optional[deleteleadPayload] = None
    add_method: Optional[str] = None
    # Flattened fields (helps when LLM doesn't nest under add/update/search)
    lead_name: Optional[str] = None
    lead_email: Optional[EmailStr] = None
    lead_phone: Optional[str] = None
    lead_id: Optional[str] = None
    confirm_delete: Optional[bool] = None
    platform: Optional[str] = None
    status: Optional[str] = None
    remarks: Optional[str] = None
    follow_up: Optional[datetime] = None
    key: Optional[str] = None
    status_type: Optional[str] = None
    insight_type: Optional[str] = Field(
        default=None,
        description="Type of insight: summary, conversion, engagement, performance, etc."
    )
    
    # Optional time filter
    date_range: Optional[str] = Field(
        default=None,
        description="Example: today, yesterday, last 7 days, this week, last month"
    )

# =============================================================================
# POSTING MANAGER
# =============================================================================

class PostingManagerPayload(BaseModel):
    platform: Optional[str] = None
    action: Optional[Literal["view_queue", "update_post", "delete_post"]] = None
    post_id: Optional[str] = None

# =============================================================================
# GENERAL TALK
# =============================================================================

class GeneralTalkPayload(BaseModel):
    message: Optional[str] = None

# =============================================================================
# TOP-LEVEL INTENT PAYLOAD
# =============================================================================

class IntentPayload(BaseModel):
    intent: Literal[
        "content_generation",
        "analytics",
        "leads_management",
        "posting_manager",
        "general_talks"
    ]

    content: Optional[ContentGenerationPayload] = None
    analytics: Optional[AnalyticsPayload] = None
    leads: Optional[LeadsManagementPayload] = None
    posting: Optional[PostingManagerPayload] = None
    general: Optional[GeneralTalkPayload] = None

# Rebuild forward references
IntentPayload.model_rebuild()
ContentGenerationPayload.model_rebuild()

# =============================================================================
# LANGGRAPH STATE
# =============================================================================

class IntentBasedChatbotState(TypedDict):
    """State for the intent-based chatbot conversation"""
    user_id: str
    current_query: str
    conversation_history: Optional[List[Dict[str, str]]]
    intent_payload: Optional[IntentPayload]  # The classified payload
    partial_payload: Optional[Dict[str, Any]]  # Accumulated partial payload data
    response: Optional[str]
    context: Dict[str, Any]
    needs_clarification: Optional[bool]  # Whether we're waiting for user input
    options: Optional[List[str]]  # Clickable options for user selection
    content_data: Optional[Dict[str, Any]] 
    last_clarification_field: Optional[str]
 # Structured content data (title, content, hashtags, images)

# =============================================================================
# INTENT-BASED CHATBOT CLASS
# =============================================================================

class IntentBasedChatbot:
    def __init__(self):
        self.llm = llm
        # Initialize token tracker for usage tracking
        supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_url and supabase_service_key:
            self.token_tracker = TokenUsageService(supabase_url, supabase_service_key)
        else:
            self.token_tracker = None
        self.setup_graph()
    
    def setup_graph(self):
        """Setup the LangGraph workflow"""
        workflow = StateGraph(IntentBasedChatbotState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("handle_content_generation", self.handle_content_generation)
        workflow.add_node("handle_analytics", self.handle_analytics)
        workflow.add_node("handle_leads_management", self.handle_leads_management)
        workflow.add_node("handle_posting_manager", self.handle_posting_manager)
        workflow.add_node("handle_general_talks", self.handle_general_talks)
        workflow.add_node("generate_final_response", self.generate_final_response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "content_generation": "handle_content_generation",
                "analytics": "handle_analytics",
                "leads_management": "handle_leads_management",
                "posting_manager": "handle_posting_manager",
                "general_talks": "handle_general_talks"
            }
        )
        
        # All handlers go to generate_final_response
        workflow.add_edge("handle_content_generation", "generate_final_response")
        workflow.add_edge("handle_analytics", "generate_final_response")
        workflow.add_edge("handle_leads_management", "generate_final_response")
        workflow.add_edge("handle_posting_manager", "generate_final_response")
        workflow.add_edge("handle_general_talks", "generate_final_response")
        workflow.add_edge("generate_final_response", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def route_by_intent(self, state: IntentBasedChatbotState) -> str:
        """Route to appropriate handler based on intent"""
        if not state.get("intent_payload"):
            return "general_talks"
        return state["intent_payload"].intent
    
    def _normalize_payload(self, payload_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and fix payload structure before validation"""
        # Fix content_generation payload if type is missing or null
        if payload_dict.get("intent") == "content_generation" and payload_dict.get("content"):
            content = payload_dict["content"]
            
            if isinstance(content, dict):
                # Check if type is missing, None, or null
                content_type = content.get("type")
                
                # Handle None/null values
                if content_type is None or content_type == "null" or content_type == "":
                    content_type = None
                
                # If type is missing or null, try to infer it
                if not content_type:
                    # Try to infer type from existing nested objects
                    if content.get("social_media"):
                        content["type"] = "social_media"
                        content_type = "social_media"
                    elif content.get("blog"):
                        content["type"] = "blog"
                        content_type = "blog"
                    elif content.get("email"):
                        content["type"] = "email"
                        content_type = "email"
                    elif content.get("whatsapp"):
                        content["type"] = "whatsapp"
                        content_type = "whatsapp"
                    elif content.get("ads"):
                        content["type"] = "ads"
                        content_type = "ads"
                    else:
                        # Check for blog-like fields (topic, length, style) - these suggest blog type
                        if any(key in content for key in ["topic", "length", "style"]):
                            # These fields suggest it might be a blog, but we need to restructure
                            # For now, default to social_media as it's most common
                            content["type"] = "social_media"
                            content_type = "social_media"
                            logger.warning("Content type missing, detected blog-like fields but defaulting to social_media")
                        else:
                            # Default to social_media if we can't infer (most common use case)
                            # This handles the case where user says "create content" without specifying type
                            content["type"] = "social_media"
                            content_type = "social_media"
                            logger.warning("Content type missing or null, defaulting to social_media")
                
                # If we have social_media type, ensure social_media nested object exists
                if content_type == "social_media":
                    if "social_media" not in content:
                        content["social_media"] = {}
                    
                    social_media = content["social_media"]
                    if not isinstance(social_media, dict):
                        social_media = {}
                        content["social_media"] = social_media
                    
                    # Check if platform or content_type are at the wrong level (directly under content)
                    # Move them to social_media if found
                    if "platform" in content and "platform" not in social_media:
                        social_media["platform"] = content.pop("platform")
                        logger.info(f"Moved platform from content to social_media: {social_media.get('platform')}")
                    
                    if "content_type" in content and "content_type" not in social_media:
                        social_media["content_type"] = content.pop("content_type")
                        logger.info(f"Moved content_type from content to social_media: {social_media.get('content_type')}")
                    
                    if "idea" in content and "idea" not in social_media:
                        social_media["idea"] = content.pop("idea")
                        logger.info(f"Moved idea from content to social_media: {social_media.get('idea')}")
                    
                    # Move media and media_file fields if they're at the wrong level
                    if "media" in content and "media" not in social_media:
                        social_media["media"] = content.pop("media")
                        logger.info(f"Moved media from content to social_media: {social_media.get('media')}")
                    
                    if "media_file" in content and "media_file" not in social_media:
                        social_media["media_file"] = content.pop("media_file")
                        logger.info(f"Moved media_file from content to social_media: {social_media.get('media_file')}")
                    
                    # Move date and task fields if they're at the wrong level
                    if "date" in content and "date" not in social_media:
                        social_media["date"] = content.pop("date")
                        logger.info(f"Moved date from content to social_media: {social_media.get('date')}")
                    
                    if "task" in content and "task" not in social_media:
                        social_media["task"] = content.pop("task")
                        logger.info(f"Moved task from content to social_media: {social_media.get('task')}")
                    
                    # Move content field (for saved_content_id) if it's at the wrong level
                    if "content" in content and "content" not in social_media:
                        social_media["content"] = content.pop("content")
                        logger.info(f"Moved content (saved_content_id) from content to social_media: {social_media.get('content')}")
                
                # If we have email type, ensure email nested object exists and normalize field names
                elif content_type == "email":
                    if "email" not in content:
                        content["email"] = {}
                    
                    email = content["email"]
                    if not isinstance(email, dict):
                        email = {}
                        content["email"] = email
                    
                    # Map common field name variations to the correct field name
                    # LLM might use "recipient" but EmailPayload uses "email_address"
                    if "recipient" in email and "email_address" not in email:
                        email["email_address"] = email.pop("recipient")
                        logger.info(f"Mapped recipient to email_address: {email.get('email_address')}")
                    
                    # Also check if recipient is at the wrong level (directly under content)
                    if "recipient" in content and "email_address" not in email:
                        email["email_address"] = content.pop("recipient")
                        logger.info(f"Moved recipient from content to email.email_address: {email.get('email_address')}")
                    
                    # Map common email content field name variations
                    # LLM might use "body", "message", "text", "subject" but EmailPayload uses "content"
                    content_field_aliases = ["body", "message", "text", "subject", "topic", "about"]
                    for alias in content_field_aliases:
                        if alias in email and "content" not in email:
                            email["content"] = email.pop(alias)
                            logger.info(f"Mapped {alias} to content: {email.get('content')[:50] if email.get('content') else None}")
                            break
                        elif alias in content and "content" not in email:
                            email["content"] = content.pop(alias)
                            logger.info(f"Moved {alias} from content to email.content: {email.get('content')[:50] if email.get('content') else None}")
                            break
                    
                    # Handle attachment/attachments fields
                    # EmailPayload uses "attachments" (plural, List[str])
                    # LLM might use "attachment" (singular) - convert to list if needed
                    if "attachment" in email and "attachments" not in email:
                        attachment_value = email.pop("attachment")
                        # Convert to list if it's a string
                        if isinstance(attachment_value, str):
                            email["attachments"] = [attachment_value]
                        elif isinstance(attachment_value, list):
                            email["attachments"] = attachment_value
                        else:
                            email["attachments"] = [str(attachment_value)] if attachment_value else []
                        logger.info(f"Mapped attachment to attachments: {email.get('attachments')}")
                    
                    # Also check if attachment/attachments is at the wrong level (directly under content)
                    if "attachment" in content and "attachments" not in email:
                        attachment_value = content.pop("attachment")
                        if isinstance(attachment_value, str):
                            email["attachments"] = [attachment_value]
                        elif isinstance(attachment_value, list):
                            email["attachments"] = attachment_value
                        else:
                            email["attachments"] = [str(attachment_value)] if attachment_value else []
                        logger.info(f"Moved attachment from content to email.attachments: {email.get('attachments')}")
                    
                    if "attachments" in content and "attachments" not in email:
                        email["attachments"] = content.pop("attachments")
                        logger.info(f"Moved attachments from content to email.attachments: {email.get('attachments')}")
                
                # If we have blog type, ensure blog nested object exists and normalize field names
                elif content_type == "blog":
                    if "blog" not in content:
                        content["blog"] = {}
                    
                    blog = content["blog"]
                    if not isinstance(blog, dict):
                        blog = {}
                        content["blog"] = blog
                    
                    # Check if blog fields are at the wrong level (directly under content)
                    # Move them to blog if found
                    blog_field_mappings = {
                        "topic": "topic",
                        "platform": "platform",
                        "length": "length",
                        "media": "media",
                        "media_file": "media_file",
                        "task": "task"
                    }
                    for field_name, target_field in blog_field_mappings.items():
                        if field_name in content and target_field not in blog:
                            blog[target_field] = content.pop(field_name)
                            logger.info(f"Moved {field_name} from content to blog.{target_field}: {blog.get(target_field)}")
                
                # If we have whatsapp type, ensure whatsapp nested object exists and normalize field names
                elif content_type == "whatsapp":
                    if "whatsapp" not in content:
                        content["whatsapp"] = {}
                    
                    whatsapp = content["whatsapp"]
                    if not isinstance(whatsapp, dict):
                        whatsapp = {}
                        content["whatsapp"] = whatsapp
                    
                    # Map common field name variations
                    # LLM might use "phone" or "number" but WhatsAppPayload uses "phone_number"
                    if "phone" in whatsapp and "phone_number" not in whatsapp:
                        whatsapp["phone_number"] = whatsapp.pop("phone")
                        logger.info(f"Mapped phone to phone_number: {whatsapp.get('phone_number')}")
                    
                    if "number" in whatsapp and "phone_number" not in whatsapp:
                        whatsapp["phone_number"] = whatsapp.pop("number")
                        logger.info(f"Mapped number to phone_number: {whatsapp.get('phone_number')}")
                    
                    # Also check if phone/number is at the wrong level (directly under content)
                    if "phone" in content and "phone_number" not in whatsapp:
                        whatsapp["phone_number"] = content.pop("phone")
                        logger.info(f"Moved phone from content to whatsapp.phone_number: {whatsapp.get('phone_number')}")
                    
                    if "number" in content and "phone_number" not in whatsapp:
                        whatsapp["phone_number"] = content.pop("number")
                        logger.info(f"Moved number from content to whatsapp.phone_number: {whatsapp.get('phone_number')}")
                    
                    # Map common message field name variations
                    # LLM might use "message", "text", "content" but WhatsAppPayload uses "text"
                    message_field_aliases = ["message", "content", "body"]
                    for alias in message_field_aliases:
                        if alias in whatsapp and "text" not in whatsapp:
                            whatsapp["text"] = whatsapp.pop(alias)
                            logger.info(f"Mapped {alias} to text: {whatsapp.get('text')[:50] if whatsapp.get('text') else None}")
                            break
                        elif alias in content and "text" not in whatsapp:
                            whatsapp["text"] = content.pop(alias)
                            logger.info(f"Moved {alias} from content to whatsapp.text: {whatsapp.get('text')[:50] if whatsapp.get('text') else None}")
                            break
                    
                    # Handle attachment field
                    # WhatsAppPayload uses "attachment" (singular, str)
                    if "attachment" in content and "attachment" not in whatsapp:
                        whatsapp["attachment"] = content.pop("attachment")
                        logger.info(f"Moved attachment from content to whatsapp.attachment: {whatsapp.get('attachment')}")
                
                # Re-check content_type after inference (it should be set by now)
                content_type = content.get("type")
                logger.info(f"Content type after inference: {content_type}")
                
                # Ensure type is valid (handle case where type was set but invalid)
                if content_type and content_type not in [None, "null", ""]:
                    valid_types = ["social_media", "blog", "email", "whatsapp", "ads"]
                    if content_type not in valid_types:
                        logger.warning(f"Invalid content type '{content_type}', defaulting to social_media")
                        content["type"] = "social_media"
                        content_type = "social_media"
                    
                    # If type is set but the corresponding nested object doesn't exist, create an empty one
                    # This ensures the payload structure is valid for validation
                    if content_type == "social_media" and "social_media" not in content:
                        content["social_media"] = {}
                    elif content_type == "blog" and "blog" not in content:
                        content["blog"] = {}
                    elif content_type == "email" and "email" not in content:
                        content["email"] = {}
                    elif content_type == "whatsapp" and "whatsapp" not in content:
                        content["whatsapp"] = {}
                    elif content_type == "ads" and "ads" not in content:
                        content["ads"] = {}
                else:
                    # If type is still None/empty after all attempts, remove the content object entirely
                    # We'll ask the user for the type in the handler
                    logger.warning(f"Content type could not be determined (type={content_type}), removing content from payload")
                    payload_dict["content"] = None
                    # Return early since we've removed content
                    return payload_dict
                
                # Remove invalid fields that don't belong in ContentGenerationPayload
                # ContentGenerationPayload should only have: type, social_media, blog, email, whatsapp, ads
                # Only do this if content still exists and has a valid type
                if content and isinstance(content, dict) and content.get("type"):
                    valid_content_keys = ["type", "social_media", "blog", "email", "whatsapp", "ads"]
                    invalid_keys = [key for key in content.keys() if key not in valid_content_keys]
                    if invalid_keys:
                        logger.warning(f"Removing invalid fields from content payload: {invalid_keys}")
                        for key in invalid_keys:
                            content.pop(key, None)
                elif content and isinstance(content, dict) and not content.get("type"):
                    # If content exists but type is still None, remove it
                    logger.warning("Content object exists but type is None, removing content from payload")
                    payload_dict["content"] = None
        
        # Normalize leads payload to keep action and fields consistent
        if payload_dict.get("intent") == "leads_management":
            leads_payload = payload_dict.get("leads")

            # Ensure leads is a dict we can work with
            if leads_payload is None or not isinstance(leads_payload, dict):
                leads_payload = leads_payload if isinstance(leads_payload, dict) else {}
                payload_dict["leads"] = leads_payload

            # Normalize field aliases (e.g., "name" -> "lead_name", "phone_number" -> "lead_phone")
            field_aliases = {
                "name": "lead_name",
                "email": "lead_email",
                "phone": "lead_phone",
                "id": "lead_id",
            }
            for alias, canonical_field in field_aliases.items():
                if leads_payload.get(alias) is not None and leads_payload.get(canonical_field) is None:
                    leads_payload[canonical_field] = leads_payload.pop(alias)
                elif payload_dict.get(alias) is not None:
                    # Only set if canonical field doesn't exist or is None/empty
                    if not leads_payload.get(canonical_field):
                        leads_payload[canonical_field] = payload_dict.pop(alias)
            
            # Normalize platform aliases (e.g., "source_platform", "source" -> "platform")
            platform_aliases = ["source_platform", "source", "platform_source", "source platform"]
            for alias in platform_aliases:
                if leads_payload.get(alias) is not None and not leads_payload.get("platform"):
                    leads_payload["platform"] = leads_payload.pop(alias)
                    logger.info(f"Normalized platform alias '{alias}' to 'platform' in _normalize_payload")
                elif payload_dict.get(alias) is not None and not leads_payload.get("platform"):
                    leads_payload["platform"] = payload_dict.pop(alias)
                    logger.info(f"Normalized platform alias '{alias}' to 'platform' from root level")

            # Move any root-level lead fields into the leads payload
            lead_fields = [
                "action",
                "lead_name",
                "lead_email",
                "lead_phone",
                "lead_id",
                "platform",
                "status",
                "remarks",
                "follow_up",
                "date_range",
                "key",
                "status_type",
                "add_method",
            ]
            for field in lead_fields:
                if payload_dict.get(field) is not None:
                    leads_payload.setdefault(field, payload_dict.pop(field))

            # Normalize action strings (e.g., "add lead" -> "add_lead")
            action = leads_payload.get("action")
            if isinstance(action, str):
                normalized_action = action.strip().lower().replace(" ", "_")
                action_aliases = {
                    "add": "add_lead",
                    "addlead": "add_lead",
                    "update": "update_lead",
                    "search": "search_lead",
                }
                leads_payload["action"] = action_aliases.get(normalized_action, normalized_action)

            add_method = leads_payload.get("add_method")
            if isinstance(add_method, str):
                normalized_method = add_method.strip().lower()
                if "import" in normalized_method or "csv" in normalized_method:
                    leads_payload["add_method"] = "import_csv"
                else:
                    leads_payload["add_method"] = "manual"
            elif add_method is not None and not isinstance(add_method, str):
                leads_payload.pop("add_method", None)

            # If no action yet, infer from nested blocks
            if not leads_payload.get("action"):
                if leads_payload.get("add_lead"):
                    leads_payload["action"] = "add_lead"
                elif leads_payload.get("update_lead"):
                    leads_payload["action"] = "update_lead"
                elif leads_payload.get("search_lead"):
                    leads_payload["action"] = "search_lead"
            # Flatten nested structures into top-level convenience fields
            if leads_payload.get("add_lead") and isinstance(leads_payload["add_lead"], dict):
                add_block = leads_payload["add_lead"]
                # Normalize platform aliases in nested block first
                platform_aliases = ["source_platform", "source", "platform_source", "source platform"]
                for alias in platform_aliases:
                    if add_block.get(alias) is not None and not add_block.get("platform"):
                        add_block["platform"] = add_block.pop(alias)
                        logger.info(f"Normalized platform alias '{alias}' to 'platform' in add_block")
                
                # Copy values from nested block, but only if existing value is None or empty
                # This preserves existing values and prevents overwriting
                if "lead_name" in add_block and add_block["lead_name"] is not None:
                    if not leads_payload.get("lead_name"):
                        leads_payload["lead_name"] = add_block["lead_name"]
                if "lead_email" in add_block and add_block["lead_email"] is not None:
                    if not leads_payload.get("lead_email"):
                        leads_payload["lead_email"] = add_block["lead_email"]
                if "lead_phone" in add_block and add_block["lead_phone"] is not None:
                    if not leads_payload.get("lead_phone"):
                        leads_payload["lead_phone"] = add_block["lead_phone"]
                if "platform" in add_block and add_block["platform"] is not None:
                    if not leads_payload.get("platform"):
                        leads_payload["platform"] = add_block["platform"]
                if "status" in add_block and add_block["status"] is not None:
                    if not leads_payload.get("status"):
                        leads_payload["status"] = add_block["status"]
                if "lead_id" in add_block and add_block["lead_id"] is not None:
                    if not leads_payload.get("lead_id"):
                        leads_payload["lead_id"] = add_block["lead_id"]
                remarks_value = add_block.get("remarks")
                # Handle "na" for optional remarks - convert to None
                if remarks_value and isinstance(remarks_value, str) and remarks_value.lower().strip() == "na":
                    leads_payload["remarks"] = None
                elif remarks_value is not None:
                    # Only set if not already set
                    if "remarks" not in leads_payload or leads_payload.get("remarks") is None:
                        leads_payload["remarks"] = remarks_value
                if "follow_up" in add_block and add_block["follow_up"] is not None:
                    if not leads_payload.get("follow_up"):
                        leads_payload["follow_up"] = add_block["follow_up"]
            
            # Normalize "na" for remarks if provided at top level
            if leads_payload.get("remarks"):
                remarks_value = leads_payload.get("remarks")
                if remarks_value and isinstance(remarks_value, str) and remarks_value.lower().strip() == "na":
                    leads_payload["remarks"] = None

            if leads_payload.get("update_lead") and isinstance(leads_payload["update_lead"], dict):
                upd_block = leads_payload["update_lead"]
                leads_payload.setdefault("lead_name", upd_block.get("lead_name"))
                leads_payload.setdefault("lead_email", upd_block.get("lead_email"))
                leads_payload.setdefault("lead_phone", upd_block.get("lead_phone"))
                leads_payload.setdefault("lead_id", upd_block.get("lead_id"))
                leads_payload.setdefault("remarks", upd_block.get("remarks"))
                leads_payload.setdefault("follow_up", upd_block.get("follow_up"))

            if leads_payload.get("search_lead") and isinstance(leads_payload["search_lead"], dict):
                sea_block = leads_payload["search_lead"]
                leads_payload.setdefault("key", sea_block.get("key"))
                leads_payload.setdefault("lead_id", sea_block.get("lead_id"))
                leads_payload.setdefault("date_range", sea_block.get("date_range") or leads_payload.get("date_range"))

        return payload_dict
    
    def _merge_payloads(self, existing: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge new payload data into existing partial payload"""
        if not existing:
            return new.copy()
        
        merged = existing.copy()
        
        # Recursively merge nested dictionaries
        for key, value in new.items():
            if value is not None:  # Only merge non-null values
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_payloads(merged[key], value)
                elif key == "leads" and isinstance(value, dict) and isinstance(merged.get(key), dict):
                    # Merge leads dict more carefully - preserve existing non-empty values
                    # First, normalize platform aliases in the new value before merging
                    platform_aliases = ["source_platform", "source", "platform_source", "source platform"]
                    for alias in platform_aliases:
                        if value.get(alias) is not None and not value.get("platform"):
                            value["platform"] = value.pop(alias)
                    
                    # Now merge the normalized value
                    for leads_key, leads_value in value.items():
                        # CRITICAL: For clarification fields, ALWAYS accept new values (even if existing)
                        # This prevents looping when user answers questions
                        clarification_fields = ["platform", "status", "lead_name", "lead_email", "lead_phone", "remarks", "action"]
                        
                        if leads_key in clarification_fields:
                            # ALWAYS update clarification fields, even None values (for tracking)
                            # But only if the new value is actually different or we're answering a question
                            existing_value = merged[key].get(leads_key)
                            if leads_value != existing_value or leads_value is not None:
                                merged[key][leads_key] = leads_value
                                logger.info(f"✅ Updated clarification field '{leads_key}': {existing_value} → {leads_value}")
                        elif leads_value is not None and leads_value != "":
                            # For other fields, normal merge logic
                            existing_value = merged[key].get(leads_key)
                            if existing_value is None or existing_value == "":
                                merged[key][leads_key] = leads_value
                            elif isinstance(existing_value, dict) and isinstance(leads_value, dict):
                                merged[key][leads_key] = self._merge_payloads(existing_value, leads_value)
                else:
                    # For other keys, only overwrite if existing is None or empty
                    if key not in merged or merged[key] is None or merged[key] == "":
                        merged[key] = value
        
        return merged
    
    def _get_missing_fields_for_social_media(self, payload: Any) -> List[Dict[str, Any]]:
        """Get list of missing required fields for social media payload"""
        missing = []
        
        if not payload:
            return [{
                "field": "platform", 
                "question": "Which platform(s) would you like to create content for?", 
                "options": ["facebook", "instagram", "youtube", "linkedin", "twitter", "pinterest"],
                "priority": 1
            }]
        
        # Required fields in priority order
        if not payload.platform:
            missing.append({
                "field": "platform",
                "question": "Which platform(s) would you like to create content for?",
                "options": ["facebook", "instagram", "youtube", "linkedin", "twitter", "pinterest"],
                "priority": 1
            })
        
        if not payload.content_type:
            missing.append({
                "field": "content_type",
                "question": "What type of content would you like to create?",
                "options": ["post", "reel", "video", "story", "carousel"],
                "priority": 2
            })
        
        if not payload.idea:
            missing.append({
                "field": "idea",
                "question": "What would you like to share in this social media post?",
                "options": None,
                "priority": 3
            })
        
        # Sort by priority
        missing.sort(key=lambda x: x.get("priority", 999))
        return missing
    
    def _extract_lead_data_from_query(self, query: str) -> Dict[str, Any]:
        """Extract lead data directly from user query using pattern matching (no LLM)
        
        DEPRECATION NOTE (2024-12-17): This function is no longer used for initial lead extraction.
        The LLM now handles initial extraction with comprehensive rules (see classify_intent prompt).
        This function is ONLY used as a fallback for extracting single-field clarification responses
        in handle_leads_management (lines 1966-2027).
        """
        import re
        
        query_lower = query.lower()
        extracted = {}
        
        # Detect action
        if any(keyword in query_lower for keyword in ["add lead", "add_lead", "create lead", "new lead", "add a lead"]):
            extracted["action"] = "add_lead"
        elif any(keyword in query_lower for keyword in ["update lead", "update_lead", "modify lead", "change lead", "edit lead"]):
            extracted["action"] = "update_lead"
        elif any(keyword in query_lower for keyword in ["search lead", "search_lead", "find lead", "look for lead"]):
            extracted["action"] = "search_lead"
        elif any(keyword in query_lower for keyword in ["delete lead", "delete_lead", "remove lead", "delete a lead"]):
            extracted["action"] = "delete_lead"
        elif any(keyword in query_lower for keyword in ["insight", "lead analytics", "lead stats"]):
            extracted["action"] = "insight"
        
        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', query)
        if email_match:
            extracted["lead_email"] = email_match.group(0)
            logger.info(f"✅ Pre-extracted lead_email: {extracted['lead_email']}")
        
        # Extract phone
        phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)\.]{6,}[0-9]', query)
        if phone_match:
            extracted["lead_phone"] = phone_match.group(0).strip()
            logger.info(f"✅ Pre-extracted lead_phone: {extracted['lead_phone']}")
        
        # Extract platform/source
        platforms = ["website", "facebook", "instagram", "linkedin", "referral", "manual", "google", "twitter", "other"]
        for platform in platforms:
            if platform in query_lower:
                extracted["platform"] = platform
                logger.info(f"✅ Pre-extracted platform: {extracted['platform']}")
                break
        
        # Extract status
        statuses = ["new", "contacted", "responded", "qualified", "converted", "lost"]
        for status in statuses:
            if status in query_lower:
                extracted["status"] = status
                logger.info(f"✅ Pre-extracted status: {extracted['status']}")
                break
        
        # Extract name (more sophisticated approach)
        # Look for patterns like "add lead [Name]", "name is [Name]", "lead [Name]"
        name_patterns = [
            r'(?:add lead|create lead|new lead)\s+(?:of\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
            r'(?:name is|named|name:)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
            r'(?:lead|contact)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, query)
            if name_match:
                potential_name = name_match.group(1).strip()
                # Exclude common words that aren't names
                exclude_words = ['Lead', 'New', 'Add', 'Create', 'Update', 'With', 'And', 'The', 'From', 'Email', 'Phone']
                if potential_name and not any(word in potential_name.split() for word in exclude_words):
                    extracted["lead_name"] = potential_name
                    logger.info(f"✅ Pre-extracted lead_name: {extracted['lead_name']}")
                    break
        
        # If we didn't find a name but have action="add_lead", try a simpler approach
        # Remove email and phone from query, then look for capitalized words
        if not extracted.get("lead_name") and extracted.get("action") == "add_lead":
            cleaned_query = query
            if email_match:
                cleaned_query = cleaned_query.replace(email_match.group(0), "")
            if phone_match:
                cleaned_query = cleaned_query.replace(phone_match.group(0), "")
            
            # Look for capitalized words (potential names)
            words = cleaned_query.split()
            capitalized_words = []
            for word in words:
                if word and len(word) > 1 and word[0].isupper() and word.lower() not in [
                    'add', 'lead', 'create', 'new', 'with', 'email', 'phone', 'from', 'platform', 
                    'status', 'source', 'this', 'that', 'want', 'would', 'like'
                ]:
                    capitalized_words.append(word)
            
            if 1 <= len(capitalized_words) <= 4:
                extracted["lead_name"] = " ".join(capitalized_words)
                logger.info(f"✅ Pre-extracted lead_name (fallback): {extracted['lead_name']}")
        
        return extracted
    
    def _get_missing_fields_for_leads(self, payload: Any, asked_questions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get list of missing required fields for leads management with priorities"""
        missing: List[Dict[str, Any]] = []
        
        # Initialize asked_questions if not provided
        if asked_questions is None:
            asked_questions = {}

        # Ensure we can access attributes defensively
        action = getattr(payload, "action", None)

        if not action:
            missing.append({
                "field": "action",
                "question": "What would you like to do with leads? (add, update, search, delete, or get insights)",
                "options": ["add lead", "update lead", "search lead", "delete lead", "insight"],
                "priority": 1
            })
            return missing

        # Flatten helpers
        lead_name = getattr(payload, "lead_name", None)
        lead_id = getattr(payload, "lead_id", None)
        lead_email = getattr(payload, "lead_email", None)
        lead_phone = getattr(payload, "lead_phone", None)
        platform = getattr(payload, "platform", None)
        status = getattr(payload, "status", None)
        remarks = getattr(payload, "remarks", None)
        status_type = getattr(payload, "status_type", None)
        key = getattr(payload, "key", None) if hasattr(payload, "key") else None
        date_range = getattr(payload, "date_range", None)

        if action == "add_lead":
            add_method = getattr(payload, "add_method", None)
            if not add_method:
                missing.append({
                    "field": "add_method",
                    "question": "Would you like to add this lead manually or import a CSV file?",
                    "options": ["manual", "import csv"],
                    "priority": 0
                })
                return missing
            if not lead_name or lead_name == "":
                missing.append({
                    "field": "lead_name",
                    "question": "What is the lead's name?",
                    "options": None,
                    "priority": 1
                })
            if not lead_email or lead_email == "":
                missing.append({
                    "field": "lead_email",
                    "question": "What is the lead's email address?",
                    "options": None,
                    "priority": 2
                })
            if not lead_phone or lead_phone == "":
                missing.append({
                    "field": "lead_phone",
                    "question": "Can you share the lead's phone number (include country code if possible)?",
                    "options": None,
                    "priority": 3
                })
            if not platform or platform == "":
                missing.append({
                    "field": "platform",
                    "question": "What is the source platform for this lead?",
                    "options": ["website", "facebook", "instagram", "linkedin", "referral", "manual", "google", "twitter", "other"],
                    "priority": 3
                })
            if not status or status == "":
                missing.append({
                    "field": "status",
                    "question": "What is the initial status of this lead?",
                    "options": ["new", "contacted", "responded", "qualified", "converted", "lost"],
                    "priority": 4
                })
            # Remarks/additional info is optional - only ask if not provided or empty
            # If user enters "na", it's considered provided (will be normalized to None on execution)
            # Check if remarks is missing (None or empty) but not "na"
            remarks_str = str(remarks).lower().strip() if remarks else ""
            if remarks is None:
                # Only ask for remarks if we haven't asked before
                # This makes it truly optional - if user doesn't answer, we proceed
                if not asked_questions.get("remarks"):
                    missing.append({
                        "field": "remarks",
                        "question": "Any additional information about this lead? (You can type 'skip' or 'na' if none)",
                        "options": ["skip"],
                        "priority": 5
                    })
                # If we've already asked for remarks and user still hasn't provided it,
                # don't ask again - treat it as intentionally skipped
            # If remarks is "na" (case-insensitive), it's considered provided, so don't ask

        elif action == "update_lead":
            if not lead_id and not lead_name:
                missing.append({
                    "field": "update_lead_name",
                    "question": "Which lead would you like to update? Please provide the lead's name.",
                    "options": None,
                    "priority": 1
                })

        elif action == "search_lead":
            if not (lead_name or lead_email or lead_phone or lead_id or date_range):
                missing.append({
                    "field": "search_criteria",
                    "question": "How should I search for leads? You can give a lead name.",
                    "options": None,
                    "priority": 1
                })

        elif action == "delete_lead":
            confirm_delete = getattr(payload, "confirm_delete", None)
            if not lead_id and not lead_name:
                missing.append({
                    "field": "lead_id_delete",
                    "question": "Which lead should I delete? Please provide the lead's name.",
                    "options": None,
                    "priority": 1
                })
            if confirm_delete is not True:
                missing.append({
                    "field": "confirm_delete",
                    "question": "Are you sure you want to delete this lead? (yes/no)",
                    "options": ["yes", "no"],
                    "priority": 2
                })

        # sort by priority to keep the first item highest-priority
        missing.sort(key=lambda x: x.get("priority", 999))
        return missing
    
    def _generate_clarifying_question(self, missing_fields: List[Dict[str, Any]], intent_type: str) -> str:
        """Generate a clarifying question with options for missing fields"""
        if not missing_fields:
            return ""
        
        # Take the first missing field (highest priority)
        field_info = missing_fields[0]
        question = field_info["question"]
        
        # Add options if available
        if field_info.get("options"):
            options_text = ", ".join([f"**{opt}**" for opt in field_info["options"]])
            question += f"\n\nYou can pick from: {options_text} - or just tell me what you prefer!"
        
        return question
    
    def classify_intent(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Classify user query into intent and populate Pydantic payload"""

        # FIRST: Check if we're handling a clarification (stored in partial_payload cache)
        partial_payload = state.get("partial_payload") or {}
        if partial_payload.get("_needs_clarification") and partial_payload.get("_last_clarification_field"):
            field = partial_payload["_last_clarification_field"]
            value = state["current_query"].strip()
            leads = partial_payload.setdefault("leads", {})
            asked_questions = partial_payload.setdefault("asked_questions", {})

            logger.info(f"🎯 DIRECT CLARIFICATION - Field: '{field}', Value: '{value}' (NO LLM)")

            # Handle action clarification (this is asked in handle_leads_management when action is missing)
            if field == "action":
                v = value.strip().lower().replace(" ", "_")
                action_aliases = {
                    "add": "add_lead",
                    "addlead": "add_lead",
                    "add_lead": "add_lead",
                    "create": "add_lead",
                    "new": "add_lead",
                    "update": "update_lead",
                    "update_lead": "update_lead",
                    "edit": "update_lead",
                    "modify": "update_lead",
                    "change": "update_lead",
                    "search": "search_lead",
                    "search_lead": "search_lead",
                    "find": "search_lead",
                    "lookup": "search_lead",
                    "delete": "delete_lead",
                    "delete_lead": "delete_lead",
                    "remove": "delete_lead",
                    "insight": "insight",
                    "insights": "insight",
                    "analytics": "insight",
                    "stats": "insight",
                    "performance": "insight",
                }
                leads["action"] = action_aliases.get(v, v)
                logger.info(f"✅ Set action = '{leads['action']}'")

            elif field == "lead_name":
                leads["lead_name"] = value
                logger.info(f"✅ Set lead_name = '{value}'")

            elif field == "update_lead_name":
                # Handle update_lead name clarification (same as lead_name)
                leads["lead_name"] = value
                logger.info(f"✅ Set lead_name = '{value}' (from update_lead_name)")

            elif field == "lead_email":
                leads["lead_email"] = value
                logger.info(f"✅ Set lead_email = '{value}'")

            elif field == "lead_phone":
                leads["lead_phone"] = value
                logger.info(f"✅ Set lead_phone = '{value}'")

            elif field == "contact":
                if "@" in value:
                    leads["lead_email"] = value
                    logger.info(f"✅ Set lead_email = '{value}'")
                else:
                    leads["lead_phone"] = value
                    logger.info(f"✅ Set lead_phone = '{value}'")

            elif field == "update_lead_contact":
                # Handle update_lead contact clarification (same as contact)
                if "@" in value:
                    leads["lead_email"] = value
                    logger.info(f"✅ Set lead_email = '{value}' (from update_lead_contact)")
                else:
                    leads["lead_phone"] = value
                    logger.info(f"✅ Set lead_phone = '{value}' (from update_lead_contact)")

            elif field == "search_criteria":
                criteria = value.strip()
                criteria_lower = criteria.lower()
                phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)\.]{6,}[0-9]', criteria)
                if "@" in criteria:
                    leads["lead_email"] = criteria
                    logger.info(f"✅ Set lead_email = '{criteria}' (from search_criteria)")
                elif phone_match:
                    leads["lead_phone"] = phone_match.group(0).strip()
                    logger.info(f"✅ Set lead_phone = '{leads['lead_phone']}' (from search_criteria)")
                elif criteria and not any(q in criteria_lower for q in ["what", "how", "when", "where", "which", "why"]):
                    leads["lead_name"] = criteria
                    logger.info(f"✅ Set lead_name = '{criteria}' (from search_criteria)")
                asked_questions.pop("search_criteria", None)

            elif field == "platform":
                leads["platform"] = value.lower()
                logger.info(f"✅ Set platform = '{value.lower()}'")

            elif field == "status":
                leads["status"] = value.lower()
                logger.info(f"✅ Set status = '{value.lower()}'")

            elif field == "add_method":
                normalized = value.strip().lower()
                if "import" in normalized or "csv" in normalized:
                    leads["add_method"] = "import_csv"
                else:
                    leads["add_method"] = "manual"
                logger.info(f"✅ Set add_method = '{leads['add_method']}'")

            elif field == "remarks":
                leads["remarks"] = None if value.lower() == "na" else value
                logger.info(f"✅ Set remarks = '{leads['remarks']}'")

            elif field == "lead_id_delete":
                # Handle delete_lead name clarification (same as lead_name)
                leads["lead_name"] = value
                logger.info(f"✅ Set lead_name = '{value}' (from lead_id_delete)")

            elif field == "update_lead_field":
                # User selected which field to update (e.g., "name", "email", "phone")
                # Store the selected field so we can ask for its new value
                value_lower = value.strip().lower()
                # Map field names to canonical names
                field_mapping = {
                    "name": "update_field_name",
                    "email": "update_field_email",
                    "phone": "update_field_phone",
                    "phone_number": "update_field_phone",
                    "platform": "update_field_platform",
                    "status": "update_field_status",
                    "remarks": "update_field_remarks",
                    "follow_up": "update_field_follow_up",
                    "followup": "update_field_follow_up",
                    "follow up": "update_field_follow_up",
                }
                selected_field = field_mapping.get(value_lower)
                if selected_field:
                    # Store which field is being updated in asked_questions so handler can access it
                    # Initialize asked_questions if it doesn't exist
                    if "asked_questions" not in partial_payload:
                        partial_payload["asked_questions"] = {}
                    partial_payload["asked_questions"]["_updating_field"] = selected_field
                    partial_payload["asked_questions"].pop("update_lead_field", None)
                    logger.info(f"✅ User selected field to update: '{value_lower}' -> '{selected_field}'")
                else:
                    logger.warning(f"⚠️ Unknown update field selected: '{value}'")

            elif field.startswith("update_lead_"):
                # Handle update field value responses (e.g., "update_lead_lead_name", "update_lead_platform")
                field_name = field.replace("update_lead_", "")
                if field_name == "lead_name":
                    leads["lead_name"] = value
                    # Also set in nested block to distinguish from identifying name
                    if "update_lead" not in leads:
                        leads["update_lead"] = {}
                    leads["update_lead"]["lead_name"] = value
                    logger.info(f"✅ Set lead_name = '{value}' (for update)")
                elif field_name == "lead_email":
                    leads["lead_email"] = value
                    if "update_lead" not in leads:
                        leads["update_lead"] = {}
                    leads["update_lead"]["lead_email"] = value
                    logger.info(f"✅ Set lead_email = '{value}' (for update)")
                elif field_name == "lead_phone":
                    leads["lead_phone"] = value
                    if "update_lead" not in leads:
                        leads["update_lead"] = {}
                    leads["update_lead"]["lead_phone"] = value
                    logger.info(f"✅ Set lead_phone = '{value}' (for update)")
                elif field_name == "platform":
                    leads["platform"] = value.lower()
                    if "update_lead" not in leads:
                        leads["update_lead"] = {}
                    leads["update_lead"]["platform"] = value.lower()
                    logger.info(f"✅ Set platform = '{value.lower()}' (for update)")
                elif field_name == "status":
                    leads["status"] = value.lower()
                    # status is usually top-level but we'll set it here too
                    logger.info(f"✅ Set status = '{value.lower()}' (for update)")
                elif field_name == "remarks":
                    v = None if value.lower() == "na" else value
                    leads["remarks"] = v
                    if "update_lead" not in leads:
                        leads["update_lead"] = {}
                    leads["update_lead"]["remarks"] = v
                    logger.info(f"✅ Set remarks = '{v}' (for update)")
                elif field_name == "follow_up":
                    leads["follow_up"] = value  # Could parse as datetime if needed
                    if "update_lead" not in leads:
                        leads["update_lead"] = {}
                    leads["update_lead"]["follow_up"] = value
                    logger.info(f"✅ Set follow_up = '{value}' (for update)")
                # NOTE: _updating_field is now cleared by the tool after processing the value
                # This prevents loops when the user provides the same value as existing.

            elif field == "confirm_delete":
                v = value.strip().lower()
                if v in ["yes", "y", "confirm", "ok", "okay", "sure"]:
                    leads["confirm_delete"] = True
                else:
                    leads["confirm_delete"] = False
                logger.info(f"✅ Set confirm_delete = '{leads['confirm_delete']}'")

            partial_payload["leads"] = leads
            partial_payload["intent"] = "leads_management"
            partial_payload["asked_questions"] = asked_questions
            # Clear clarification flags from cache
            partial_payload.pop("_needs_clarification", None)
            partial_payload.pop("_last_clarification_field", None)
            
            state["partial_payload"] = partial_payload
            state["needs_clarification"] = False
            state["last_clarification_field"] = None

            state["intent_payload"] = IntentPayload(
                intent="leads_management",
                content=None,
                analytics=None,
                leads=None,
                posting=None,
                general=None,
            )
            logger.info(f"⚡ Bypassed LLM completely - handled clarification directly")
            return state
        
        query = state["current_query"]
    
        partial_payload = state.get("partial_payload")
        conversation_history = state.get("conversation_history", [])
        

        # CRITICAL: Check if we're continuing a leads_management flow
        # This bypasses LLM entirely for ALL leads clarification answers (name, email, platform, etc.)
        if partial_payload and partial_payload.get("intent") == "leads_management":
            # We're in a leads flow - skip LLM completely and go directly to handler
            state["intent_payload"] = IntentPayload(
                intent="leads_management",
                content=None,
                analytics=None,
                leads=None,
                posting=None,
                general=None,
            )
            logger.info(f"⚡ Continuing leads_management flow - skipping LLM entirely for: '{query}'")
            return state

        # Lightweight heuristic: if the user mentions leads explicitly, route to leads_management
        # BEFORE LLM classification. This prevents misrouting to general_talks when the user is
        # clearly asking about leads.
        #
        # Safety: only trigger on common lead-management phrasing to avoid false positives
        # (e.g. "lead generation" content requests).
        query_lower = query.lower()
        lead_triggers = [
            " lead", "leads", "lead ",  # basic
            "add lead", "create lead", "new lead",
            "update lead", "edit lead", "modify lead",
            "search lead", "search leads", "find lead", "find leads",
            "delete lead", "remove lead",
            "lead insights", "lead analytics", "lead stats",
            "leads page",
        ]
        # Avoid routing content_generation phrases that might mention "lead generation"
        lead_generation_exclusions = ["lead generation", "leads generation"]
        if any(t in query_lower for t in lead_triggers) and not any(x in query_lower for x in lead_generation_exclusions):
            seeded_payload = partial_payload.copy() if isinstance(partial_payload, dict) else {}
            seeded_payload["intent"] = "leads_management"
            seeded_payload.setdefault("leads", {})

            # If we can safely infer the action from the same message, set it; otherwise handler will ask.
            if not seeded_payload["leads"].get("action"):
                if any(k in query_lower for k in ["add lead", "create lead", "new lead", "add a lead"]):
                    seeded_payload["leads"]["action"] = "add_lead"
                elif any(k in query_lower for k in ["update lead", "edit lead", "modify lead", "change lead"]):
                    seeded_payload["leads"]["action"] = "update_lead"
                elif any(k in query_lower for k in ["search lead", "search leads", "find lead", "find leads", "look for lead"]):
                    seeded_payload["leads"]["action"] = "search_lead"
                elif any(k in query_lower for k in ["delete lead", "remove lead", "delete a lead"]):
                    seeded_payload["leads"]["action"] = "delete_lead"
                elif any(k in query_lower for k in ["lead insights", "insight", "analytics", "stats", "performance"]):
                    seeded_payload["leads"]["action"] = "insight"

            prefilled_leads = self._extract_lead_data_from_query(query)
            for key, value in prefilled_leads.items():
                if value is None:
                    continue
                if key == "action":
                    seeded_payload["leads"].setdefault("action", value)
                    continue
                if not seeded_payload["leads"].get(key):
                    seeded_payload["leads"][key] = value

            state["partial_payload"] = seeded_payload
            state["intent_payload"] = IntentPayload(
                intent="leads_management",
                content=None,
                analytics=None,
                leads=None,
                posting=None,
                general=None,
            )
            logger.info("✅ Pre-LLM heuristic routing to leads_management")
            return state
        
        # Log the user query
        logger.info(f"Classifying intent for query: {query}")
        if partial_payload:
            logger.info(f"Merging with existing partial payload: {json.dumps(partial_payload, indent=2)}")
        
        # Build context from conversation history
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-5:]  # Last 5 messages
            history_context = "\n\nRecent conversation:\n"
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_context += f"{role}: {content}\n"
        
        # Include partial payload context if exists
        partial_context = ""
        existing_intent = None
        if partial_payload:
            existing_intent = partial_payload.get("intent")
            partial_context = f"\n\nPreviously collected information:\n{json.dumps(partial_payload, indent=2)}\n\nCRITICAL: If there's an existing intent in the previously collected information (and it's not \"general_talks\"), you MUST preserve that exact same intent. Do NOT change the intent when the user is providing clarification answers like \"website\", \"facebook\", \"new\", etc. These are answers to questions, not new intent requests. Extract any new information from the user's query and merge it with the existing data. Keep all previously collected non-null values, including the intent."
        
        # Create the classification prompt
        classification_prompt = f"""You are an intent classifier for a business assistant chatbot.

Your job:
1. Read the user's natural language query
2. Classify it into the correct intent (one of: content_generation, analytics, leads_management, posting_manager, general_talks)
3. Extract ALL entities and information from the user's query - be thorough and extract everything mentioned
4. Produce a Pydantic-validated payload according to the provided schema
5. If the user query does not contain enough information, populate whatever fields you can and leave the rest as null
6. DO NOT hallucinate missing information
7. If information is missing and required later, mark the missing fields as null - the graph nodes will ask clarifying questions
8. Always output JSON only, following the exact structure of the Pydantic models
9. If there's existing partial payload data, merge the new information with it (keep existing non-null values, only update with new information from the current query)
10. CRITICAL: If there's an existing intent in the previously collected information (and it's not "general_talks"), you MUST preserve that exact same intent. Do NOT change the intent when the user is providing clarification answers like "website", "facebook", "new", etc. These are answers to questions, not new intent requests.

Your output MUST strictly follow this root structure:
{{
  "intent": "...",
  "content": {{"type": "social_media" | "blog" | "email" | "whatsapp" | "ads", "social_media": {{"platform": "...", "content_type": "...", "idea": "..."}}, ...}} | null,
  "analytics": {{...}} | null,
  "leads": {{...}} | null,
  "posting": {{...}} | null,
  "general": {{...}} | null
}}

CRITICAL: ENTITY EXTRACTION FOR CONTENT GENERATION
When the user mentions content creation, you MUST extract ALL entities from their query:

1. PLATFORM EXTRACTION - Extract platform names from the query:
   - "instagram" → platform: ["instagram"]
   - "facebook" → platform: ["facebook"]
   - "youtube" → platform: ["youtube"]
   - "linkedin" → platform: ["linkedin"]
   - "twitter" → platform: ["twitter"]
   - "pinterest" → platform: ["pinterest"]
   - If multiple platforms mentioned, extract all: ["instagram", "facebook"]
   - If user says "instagram reel", extract platform: ["instagram"]

2. CONTENT_TYPE EXTRACTION - Extract content type from the query:
   - "reel" or "reels" → content_type: "reel"
   - "post" or "posts" → content_type: "post"
   - "video" or "videos" → content_type: "video"
   - "story" or "stories" → content_type: "story"
   - "carousel" or "carousels" → content_type: "carousel"
   - If user says "instagram reel", extract content_type: "reel"

3. IDEA/TOPIC EXTRACTION - Extract any topic, idea, or subject mentioned:
   - "product launch" → idea: "product launch"
   - "company update" → idea: "company update"
   - "tip about marketing" → idea: "tip about marketing"
   - Any descriptive text about what the content should be about

EXAMPLES OF CORRECT EXTRACTION:
- User: "i want to create an instagram reel"
  Output: {{
    "intent": "content_generation",
    "content": {{
      "type": "social_media",
      "social_media": {{
        "platform": ["instagram"],
        "content_type": "reel",
        "idea": null
      }}
    }},
    ...
  }}

- User: "create a facebook post about our new product"
  Output: {{
    "intent": "content_generation",
    "content": {{
      "type": "social_media",
      "social_media": {{
        "platform": ["facebook"],
        "content_type": "post",
        "idea": "our new product"
      }}
    }},
    ...
  }}

- User: "make a youtube video"
  Output: {{
    "intent": "content_generation",
    "content": {{
      "type": "social_media",
      "social_media": {{
        "platform": ["youtube"],
        "content_type": "video",
        "idea": null
      }}
    }},
    ...
  }}

IMPORTANT RULES FOR CONTENT GENERATION:
- If intent is "content_generation", the "content" object MUST include a "type" field
- The "type" field MUST be one of: "social_media", "blog", "email", "whatsapp", "ads"
- If the user says "post", "reel", "video", "story", "carousel" → infer type as "social_media"
- If the user says "blog" or "article", infer type as "blog"
- If the user says "email", infer type as "email"
- If the user says "whatsapp" or "message", infer type as "whatsapp"
- If the user says "ad" or "advertisement", infer type as "ads"
- ALWAYS extract platform and content_type when mentioned in the query
- NEVER create a "content" object without a "type" field
- For social_media type, ALWAYS create the "social_media" nested object with extracted fields

EMAIL-SPECIFIC RULES:
- For email type, use these EXACT field names in the "email" nested object:
  - "email_address" (NOT "recipient", "to", "email", etc.) - the recipient's email address
  - "content" (NOT "body", "message", "text", "subject", etc.) - what the email should be about
  - "attachments" (array of strings) - file paths or URLs for email attachments (extract if user mentions "attach", "attachment", "file", "document", etc.)
  - "task" (one of: "send", "save", "schedule") - what to do with the email
- If user mentions an email address, extract it as "email_address" in the "email" object
- If user describes what the email should be about (e.g., "product launch", "meeting invitation"), extract it as "content" in the "email" object
- If user mentions attachments (e.g., "attach the PDF", "with the document"), extract file references as "attachments" array
- Example: User says "send email to john@example.com about product launch"
  Output: {{"intent": "content_generation", "content": {{"type": "email", "email": {{"email_address": "john@example.com", "content": "product launch", "task": "send"}}}}}}
- Example: User says "email to jane@example.com with the invoice attached"
  Output: {{"intent": "content_generation", "content": {{"type": "email", "email": {{"email_address": "jane@example.com", "attachments": ["invoice"], "task": "send"}}}}}}

WHATSAPP-SPECIFIC RULES:
- For whatsapp type, use these EXACT field names in the "whatsapp" nested object:
  - "phone_number" (NOT "phone", "number", etc.) - recipient's phone number with country code
  - "text" (NOT "message", "content", "body", etc.) - the message text
  - "attachment" (string) - file path or URL for WhatsApp attachment (extract if user mentions "attach", "attachment", "file", "image", "video", etc.)
  - "task" (one of: "send", "schedule", "save") - what to do with the message
- If user mentions a phone number, extract it as "phone_number" in the "whatsapp" object
- If user mentions attachments (e.g., "send image", "with a video"), extract file reference as "attachment" string
- Example: User says "send WhatsApp to +919876543210 with the image"
  Output: {{"intent": "content_generation", "content": {{"type": "whatsapp", "whatsapp": {{"phone_number": "+919876543210", "attachment": "image", "task": "send"}}}}}}

BLOG-SPECIFIC RULES:
- For blog type, use these EXACT field names in the "blog" nested object:
  - "topic" - what the blog post should be about (e.g., "marketing tips", "product review")
  - "platform" (one of: "wordpress", "shopify", "wix", "html") - where to publish the blog
  - "length" (one of: "short", "medium", "long") - how long the blog post should be
  - "media" (one of: "generate", "upload") - whether to generate new media or upload existing media
  - "media_file" (string) - file path or URL if user mentions uploading a specific file
  - "task" (one of: "draft", "schedule", "save") - what to do with the blog post
- If user mentions a blog topic, extract it as "topic" in the "blog" object
- If user mentions a platform (wordpress, shopify, wix, html), extract it as "platform" in the "blog" object
- If user mentions length (short, medium, long), extract it as "length" in the "blog" object
- If user says "upload image", "use this photo", "attach file" → set media: "upload" and extract file reference as "media_file"
- If user says "generate image", "create visual", "make graphic" → set media: "generate"
- Example: User says "create a blog post about digital marketing for wordpress"
  Output: {{"intent": "content_generation", "content": {{"type": "blog", "blog": {{"topic": "digital marketing", "platform": "wordpress", "length": null, "task": null}}}}}}
- Example: User says "blog post with this image: /path/to/image.jpg"
  Output: {{"intent": "content_generation", "content": {{"type": "blog", "blog": {{"media": "upload", "media_file": "/path/to/image.jpg"}}}}}}
- When merging with existing partial payload, preserve all non-null blog fields and only update with new information

SOCIAL MEDIA MEDIA RULES:
- For social_media type, also extract media information if mentioned:
  - "media" (one of: "upload", "generate") - whether to upload existing media or generate new media
  - "media_file" (string) - file path or URL if user mentions uploading a specific file
- If user says "upload image", "use this photo", "attach file", OR JUST "upload" → set media: "upload" and extract file reference as "media_file" if provided
- If user says "generate image", "create visual", "make graphic", OR JUST "generate" → set media: "generate"
- CRITICAL: If the user responds with ONLY "generate" or ONLY "upload" (without other context), this is a direct answer to the media question - extract it as media: "generate" or media: "upload" respectively

SOCIAL MEDIA TASK AND DATE RULES:
- For social_media type, also extract task and date information if mentioned:
  - "task" (one of: "draft", "schedule", "edit", "delete") - what to do with the post after generation
  - "date" (ISO datetime string) - when to schedule the post (only needed if task is "schedule")
- If user says "draft", "save as draft", "save it" → set task: "draft"
- If user says "schedule", "schedule it", "schedule for later" → set task: "schedule"
- If user says "edit", "modify", "change" → set task: "edit"
- If user says "delete", "remove", "remove it" → set task: "delete"
- If user mentions a date/time for scheduling (e.g., "December 25, 2024 at 2:00 PM", "tomorrow at 10am"), extract it as "date" in ISO format
- CRITICAL: If the user responds with ONLY "draft", "schedule", "edit", or "delete" (without other context), this is a direct answer to the task question - extract it accordingly
- Example: User says "create an instagram post with this image: /path/to/image.jpg"
  Output: {{"intent": "content_generation", "content": {{"type": "social_media", "social_media": {{"platform": ["instagram"], "content_type": "post", "media": "upload", "media_file": "/path/to/image.jpg"}}}}}}
- Example: User says "generate" (as a response to media question)
  Output: {{"intent": "content_generation", "content": {{"type": "social_media", "social_media": {{"media": "generate"}}}}}}
- Example: User says "schedule it for tomorrow at 10am"
  Output: {{"intent": "content_generation", "content": {{"type": "social_media", "social_media": {{"task": "schedule", "date": "2024-12-26T10:00:00Z"}}}}}}
- Example: User says "edit" (as a response to task question)
  Output: {{"intent": "content_generation", "content": {{"type": "social_media", "social_media": {{"task": "edit"}}}}}}
- Example: User says "delete" (as a response to task question)
  Output: {{"intent": "content_generation", "content": {{"type": "social_media", "social_media": {{"task": "delete"}}}}}}
- When merging with existing partial payload, preserve all non-null social_media fields and only update with new information

LEADS MANAGEMENT RULES:
For leads_management intent, you MUST extract ALL lead-related information from the query.

1. ACTION EXTRACTION - Detect what the user wants to do:
   - "add lead", "create lead", "new lead", "add a lead" → action: "add_lead"
   - "update lead", "modify lead", "change lead", "edit lead" → action: "update_lead"
   - "search lead", "find lead", "look for lead" → action: "search_lead"
   - "lead insights", "lead analytics", "lead stats", "lead performance" → action: "insight"

2. LEAD DATA EXTRACTION:
   - Extract lead_name: any person's name mentioned (e.g., "John Doe", "Sarah Smith")
   - Extract lead_email: any email address (e.g., "john@example.com")
   - Extract lead_phone: any phone number with country code (e.g., "+919876543210", "123-456-7890")
   - Extract platform: where the lead came from (website, facebook, instagram, linkedin, referral, manual, google, twitter, other)
   - Extract status: lead status (new, contacted, responded, qualified, converted, lost)
   - Extract remarks: any additional notes or comments about the lead
   - Extract follow_up: any date/time mentioned for follow-up (ISO format)
   - Extract lead_id: any lead ID or identifier mentioned

3. STRUCTURED OUTPUT FOR LEADS:
   The "leads" object should follow this structure:
   {{
     "action": "add_lead" | "update_lead" | "search_lead" | "delete_lead" | "insight",
     "add_lead": {{
       "lead_name": "...",
       "lead_email": "...",
       "lead_phone": "...",
       "platform": "...",
       "status": "...",
       "remarks": "...",
       "follow_up": "ISO datetime",
       "lead_id": "..."
     }},
     "update_lead": {{
       "lead_name": "...",
       "lead_email": "...",
       "lead_phone": "...",
       "platform": "...",
       "remarks": "...",
       "follow_up": "ISO datetime",
       "lead_id": "..."
     }},
     "search_lead": {{
       "key": "search term",
       "lead_id": "...",
       "date_range": "..."
     }},
     "insight": {{
       "lead_id": "...",
       "lead_name": "...",
       "platform": "...",
       "date_range": "...",
       "insight_type": "summary | conversion | engagement | performance"
     }},
     // Flattened fields for convenience (populate these too):
     "lead_name": "...",
     "lead_email": "...",
     "lead_phone": "...",
     "platform": "...",
     "status": "...",
     "remarks": "..."
   }}

EXAMPLES OF CORRECT LEADS EXTRACTION:

- User: "add lead John Doe with email john@example.com from website"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "add_lead",
      "add_lead": {{
        "lead_name": "John Doe",
        "lead_email": "john@example.com",
        "platform": "website",
        "status": null,
        "remarks": null,
        "lead_phone": null,
        "follow_up": null,
        "lead_id": null
      }},
      "lead_name": "John Doe",
      "lead_email": "john@example.com",
      "platform": "website"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

- User: "create a new lead Sarah Smith phone +919876543210 from facebook status new"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "add_lead",
      "add_lead": {{
        "lead_name": "Sarah Smith",
        "lead_phone": "+919876543210",
        "platform": "facebook",
        "status": "new",
        "lead_email": null,
        "remarks": null,
        "follow_up": null,
        "lead_id": null
      }},
      "lead_name": "Sarah Smith",
      "lead_phone": "+919876543210",
      "platform": "facebook",
      "status": "new"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

- User: "add lead Mike Johnson mike@example.com +1234567890 from linkedin status contacted remarks: very interested in our product"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "add_lead",
      "add_lead": {{
        "lead_name": "Mike Johnson",
        "lead_email": "mike@example.com",
        "lead_phone": "+1234567890",
        "platform": "linkedin",
        "status": "contacted",
        "remarks": "very interested in our product",
        "follow_up": null,
        "lead_id": null
      }},
      "lead_name": "Mike Johnson",
      "lead_email": "mike@example.com",
      "lead_phone": "+1234567890",
      "platform": "linkedin",
      "status": "contacted",
      "remarks": "very interested in our product"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

- User: "search for leads from website"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "search_lead",
      "search_lead": {{
        "key": null,
        "lead_id": null,
        "date_range": null
      }},
      "platform": "website"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

- User: "find lead john@example.com"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "search_lead",
      "search_lead": {{
        "key": "john@example.com",
        "lead_id": null,
        "date_range": null
      }},
      "lead_email": "john@example.com"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

- User: "update lead john@example.com remarks: contacted via email"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "update_lead",
      "update_lead": {{
        "lead_email": "john@example.com",
        "remarks": "contacted via email",
        "lead_name": null,
        "lead_phone": null,
        "platform": null,
        "follow_up": null,
        "lead_id": null
      }},
      "lead_email": "john@example.com",
      "remarks": "contacted via email"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

- User: "show me insights for leads from instagram"
  Output: {{
    "intent": "leads_management",
    "leads": {{
      "action": "insight",
      "insight": {{
        "platform": "instagram",
        "date_range": null,
        "insight_type": null,
        "lead_id": null,
        "lead_name": null
      }},
      "platform": "instagram"
    }},
    "content": null,
    "analytics": null,
    "posting": null,
    "general": null
  }}

CRITICAL RULES FOR LEADS:
- Always extract ALL mentioned fields (name, email, phone, platform, status, remarks)
- Populate both nested (add_lead/update_lead/search_lead/insight) AND flattened fields
- If user provides single-word answers like "website", "new", "facebook" during clarification, extract them appropriately based on context
- Never hallucinate missing information - leave fields as null if not mentioned
- Extract phone numbers with country codes when present
- Extract email addresses in proper format
- Platform must be one of: website, facebook, instagram, linkedin, referral, manual, google, twitter, other
- Status must be one of: new, contacted, responded, qualified, converted, lost
- For action field, use exact values: add_lead, update_lead, search_lead, delete_lead, insight (with underscores, not spaces)
- When merging with existing partial payload, preserve all non-null leads fields and only update with new information

General Rules:
- EXACT intent labels must be used: content_generation, analytics, leads_management, posting_manager, general_talks
- EXACT enum values must be used (e.g., "facebook", "instagram" for platforms; "post", "reel", "video", "story", "carousel" for content_type)
- Never output fields that are not in the Pydantic schema
- Never assume unknown fields
- If a query is conversational or does not match any domain, classify it under "general_talks"
- You must always return every top-level key, even if null
- If merging with existing data, preserve all non-null existing values and only add/update with new information from the current query
- BE THOROUGH: Extract every piece of information the user mentions - don't leave fields as null if they're clearly stated in the query

{partial_context}
{history_context}

User query: "{query}"

Return ONLY valid JSON matching the IntentPayload structure. No explanations, no markdown, no comments."""

        try:
            # Use structured output to get JSON
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            
            # Log the raw LLM response
            logger.info(f"LLM raw response: {response.content}")
            
            # Parse the JSON response
            try:
                content = response.content.strip()
                
                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]  # Remove ```json
                elif content.startswith("```"):
                    content = content[3:]  # Remove ```
                
                if content.endswith("```"):
                    content = content[:-3]  # Remove closing ```
                
                content = content.strip()
                
                # Log the cleaned content before parsing
                logger.debug(f"LLM cleaned response: {content}")
                
                payload_dict = json.loads(content)
                
                # Merge with existing partial payload if it exists
                if partial_payload:
                    existing_intent = partial_payload.get("intent")
                    payload_dict = self._merge_payloads(partial_payload, payload_dict)
                    logger.info(f"Merged payload dict: {json.dumps(payload_dict, indent=2)}")
                    
                    # CRITICAL: Preserve the original intent if we're in a clarification flow
                    # This prevents the LLM from changing intent when user provides clarification answers
                    if existing_intent and existing_intent != "general_talks":
                        payload_dict["intent"] = existing_intent
                        logger.info(f"Preserved original intent '{existing_intent}' during clarification")
                
                # Log the parsed payload
                logger.info(f"Parsed payload dict: {json.dumps(payload_dict, indent=2)}")
                
                # Log lead extraction quality if this is a leads_management intent
                if payload_dict.get("intent") == "leads_management" and payload_dict.get("leads"):
                    logger.info(f"✅ LLM extracted leads payload: {json.dumps(payload_dict.get('leads'), indent=2)}")
                    leads_data = payload_dict.get("leads", {})
                    extracted_fields = [k for k, v in leads_data.items() if v is not None and k != "action"]
                    logger.info(f"📊 Lead extraction summary - Action: {leads_data.get('action')}, Fields extracted: {len(extracted_fields)} ({', '.join(extracted_fields[:10])})")
                
                # Normalize and fix payload structure (but don't validate yet)
                # IMPORTANT: Preserve intent before normalization
                preserved_intent = payload_dict.get("intent")
                payload_dict = self._normalize_payload(payload_dict)
                
                # Restore intent if it was lost during normalization
                if preserved_intent and not payload_dict.get("intent"):
                    payload_dict["intent"] = preserved_intent
                    logger.info(f"Restored intent '{preserved_intent}' after normalization")
                
                logger.info(f"Payload dict after normalization: {json.dumps(payload_dict, indent=2)}")
                
                # IMPORTANT: Ensure content is None if it's invalid (has type=None or missing type)
                # This prevents validation errors when creating minimal IntentPayload
                if payload_dict.get("content") and isinstance(payload_dict["content"], dict):
                    content_obj = payload_dict["content"]
                    content_type = content_obj.get("type")
                    if not content_type or content_type is None or content_type == "null" or content_type == "":
                        logger.warning(f"Content object has invalid type (type={content_type}), removing it from payload")
                        payload_dict["content"] = None
                
                # Store the merged payload as partial_payload for next iteration
                state["partial_payload"] = payload_dict
                
                # Create a minimal IntentPayload with just the intent for routing
                # We'll validate the full payload later when all required fields are collected
                intent_value = payload_dict.get("intent", "general_talks")
                
                # Create minimal payload for routing - only validate the intent
                # IMPORTANT: Set all payload fields to None to avoid validation
                # We only need the intent for routing
                try:
                    # Create IntentPayload with minimal data for routing
                    # Always set content to None to avoid validation
                    minimal_payload = {
                        "intent": intent_value,
                        "content": None,  # Always None - we'll validate later
                        "analytics": None,
                        "leads": None,
                        "posting": None,
                        "general": None
                    }
                    intent_payload = IntentPayload(**minimal_payload)
                    state["intent_payload"] = intent_payload
                    logger.info(f"Intent classified as: {intent_value} (payload stored in partial_payload, validation deferred)")
                except Exception as e:
                    logger.error(f"Failed to create minimal IntentPayload: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Fallback to general_talks
                    state["intent_payload"] = IntentPayload(
                        intent="general_talks",
                        general=GeneralTalkPayload(message=query)
                    )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.error(f"Response content: {response.content}")
                # Fallback to general_talks
                state["intent_payload"] = IntentPayload(
                    intent="general_talks",
                    general=GeneralTalkPayload(message=query)
                )
            except Exception as e:
                logger.error(f"Error processing payload: {e}")
                logger.error(f"Payload dict: {json.dumps(payload_dict, indent=2) if 'payload_dict' in locals() else 'N/A'}")
                
                # Try to fix common issues and store as partial payload
                if 'payload_dict' in locals():
                    try:
                        # Try normalization again
                        fixed_payload = self._normalize_payload(payload_dict.copy())
                        state["partial_payload"] = fixed_payload
                        
                        # Create minimal IntentPayload for routing
                        intent_value = fixed_payload.get("intent", "general_talks")
                        minimal_payload = {
                            "intent": intent_value,
                            "content": None,
                            "analytics": None,
                            "leads": None,
                            "posting": None,
                            "general": None
                        }
                        intent_payload = IntentPayload(**minimal_payload)
                        state["intent_payload"] = intent_payload
                        logger.info(f"Fixed payload structure, intent: {intent_value} (validation deferred)")
                    except Exception as retry_error:
                        logger.error(f"Retry after normalization also failed: {retry_error}")
                        # Fallback to general_talks
                        state["intent_payload"] = IntentPayload(
                            intent="general_talks",
                            general=GeneralTalkPayload(message=query)
                        )
                else:
                    # Fallback to general_talks
                    state["intent_payload"] = IntentPayload(
                        intent="general_talks",
                        general=GeneralTalkPayload(message=query)
                    )
                
        except Exception as e:
            logger.error(f"Error in classify_intent: {e}")
            # Fallback to general_talks
            state["intent_payload"] = IntentPayload(
                intent="general_talks",
                general=GeneralTalkPayload(message=query)
            )
        
        return state
    
    def handle_content_generation(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Handle content generation intent"""
        try:
            from agents.tools.Leo_Content_Generation import execute_content_generation
            
            # Get the partial payload dictionary (not validated yet)
            partial_payload = state.get("partial_payload", {})
            content_dict = partial_payload.get("content")
            
            # Quick check: if user just said "generate" or "upload" and we're waiting for media, set it directly
            # Also check for "draft" or "schedule" for task field
            user_query = state.get("current_query", "").strip()
            user_query_lower = user_query.lower()
            if content_dict and content_dict.get("type") == "social_media":
                social_media_dict = content_dict.get("social_media", {})
                
                # Check if user sent "upload {url}" format (from file upload)
                if user_query_lower.startswith("upload ") and len(user_query) > 7:
                    # Extract URL from "upload {url}"
                    file_url = user_query[7:].strip()  # Remove "upload " prefix
                    if file_url and (file_url.startswith("http://") or file_url.startswith("https://")):
                        if "social_media" not in content_dict:
                            content_dict["social_media"] = {}
                        content_dict["social_media"]["media"] = "upload"
                        content_dict["social_media"]["media_file"] = file_url
                        partial_payload["content"] = content_dict
                        state["partial_payload"] = partial_payload
                        logger.info(f"Directly set media to 'upload' with file URL: {file_url}")
                elif not social_media_dict.get("media"):
                    if user_query_lower == "generate":
                        if "social_media" not in content_dict:
                            content_dict["social_media"] = {}
                        content_dict["social_media"]["media"] = "generate"
                        partial_payload["content"] = content_dict
                        state["partial_payload"] = partial_payload
                        logger.info("Directly set media to 'generate' from user query")
                    elif user_query_lower == "upload":
                        if "social_media" not in content_dict:
                            content_dict["social_media"] = {}
                        content_dict["social_media"]["media"] = "upload"
                        partial_payload["content"] = content_dict
                        state["partial_payload"] = partial_payload
                        logger.info("Directly set media to 'upload' from user query")
            
            # Re-get content_dict after potential update
            content_dict = partial_payload.get("content")
            
            if not content_dict:
                state["response"] = "I'd love to help you create some content! What are you thinking - are you looking to create something for social media, write a blog post, send an email, create a WhatsApp message, or maybe work on some ads?"
                state["needs_clarification"] = True
                return state
            
            # Check if type is set
            content_type = content_dict.get("type")
            if not content_type:
                state["response"] = "Sounds good! Just to make sure I create exactly what you need - are you thinking social media content, a blog post, an email, a WhatsApp message, or maybe some ads?"
                state["needs_clarification"] = True
                return state
            
            # Check for missing required fields based on content type
            # Handle social_media type
            if content_type == "social_media":
                social_media_dict = content_dict.get("social_media", {})
                
                # Check for missing fields using dictionary
                missing_fields = []
                if not social_media_dict.get("platform"):
                    missing_fields.append({
                        "field": "platform",
                        "question": "Great! Which social media platform are you thinking of? Are you looking to post on Facebook, Instagram, YouTube, LinkedIn, Twitter, or Pinterest?",
                        "options": ["facebook", "instagram", "youtube", "linkedin", "twitter", "pinterest"],
                        "priority": 1
                    })
                
                if not social_media_dict.get("content_type"):
                    missing_fields.append({
                        "field": "content_type",
                        "question": "What kind of content are you planning? Are you thinking of a regular post, a reel, a video, a story, or maybe a carousel?",
                        "options": ["post", "reel", "video", "story", "carousel"],
                        "priority": 2
                    })
                
                if not social_media_dict.get("idea"):
                    missing_fields.append({
                        "field": "idea",
                        "question": "What would you like to share in this social media post?",
                        "options": None,
                        "priority": 3
                    })
                
                # Check for media field (lower priority - ask after core fields are filled)
                if not social_media_dict.get("media"):
                    missing_fields.append({
                        "field": "media",
                        "question": "Great! For the visuals, would you like me to generate an image for this post, or do you have a file you'd like to upload?",
                        "options": ["generate", "upload"],
                        "priority": 4
                    })
                # If media is "upload" but media_file is missing, ask for the file
                elif social_media_dict.get("media") == "upload" and not social_media_dict.get("media_file"):
                    missing_fields.append({
                        "field": "media_file",
                        "question": "Perfect! You mentioned uploading a file. Could you share the file path or URL for the image/video you'd like to use?",
                        "options": None,
                        "priority": 4
                    })
                
                # Sort by priority
                missing_fields.sort(key=lambda x: x.get("priority", 999))
                
                logger.info(f"Missing fields for social_media: {missing_fields}")
                if missing_fields:
                    question = self._generate_clarifying_question(missing_fields, "social_media")
                    state["response"] = question
                    state["needs_clarification"] = True
                    # Store options for frontend rendering
                    field_info = missing_fields[0]
                    state["options"] = field_info.get("options")
                    logger.info(f"Generated clarifying question: {question}")
                    logger.info(f"Options for frontend: {state['options']}")
                    return state
            
            # Handle email type
            elif content_type == "email":
                email_dict = content_dict.get("email", {})
                
                # Check for missing fields using dictionary
                # Also check for "recipient" as an alias for "email_address"
                missing_fields = []
                email_address = email_dict.get("email_address") or email_dict.get("recipient")
                if not email_address:
                    missing_fields.append({
                        "field": "email_address",
                        "question": "Sure! Who should I send this email to? What's their email address?",
                        "options": None,
                        "priority": 1
                    })
                
                # Check for content field, also check common aliases
                email_content = email_dict.get("content") or email_dict.get("body") or email_dict.get("message") or email_dict.get("text") or email_dict.get("subject") or email_dict.get("topic") or email_dict.get("about")
                if not email_content:
                    missing_fields.append({
                        "field": "content",
                        "question": "What's this email going to be about? Are you announcing a product, inviting them to a meeting, sending a newsletter, or something else?",
                        "options": None,
                        "priority": 2
                    })
                
                if not email_dict.get("task"):
                    missing_fields.append({
                        "field": "task",
                        "question": "Got it! What would you like me to do with this email? Should I send it right away, save it as a draft, or schedule it for later?",
                        "options": ["send", "save", "schedule"],
                        "priority": 3
                    })
                
                # Sort by priority
                missing_fields.sort(key=lambda x: x.get("priority", 999))
                
                logger.info(f"Missing fields for email: {missing_fields}")
                if missing_fields:
                    question = self._generate_clarifying_question(missing_fields, "email")
                    state["response"] = question
                    state["needs_clarification"] = True
                    # Store options for frontend rendering
                    field_info = missing_fields[0]
                    state["options"] = field_info.get("options")
                    logger.info(f"Generated clarifying question: {question}")
                    logger.info(f"Options for frontend: {state['options']}")
                    return state
            
            # Handle blog type
            elif content_type == "blog":
                blog_dict = content_dict.get("blog", {})
                
                # Check for missing fields using dictionary
                missing_fields = []
                if not blog_dict.get("topic"):
                    missing_fields.append({
                        "field": "topic",
                        "question": "Awesome! What topic are you thinking of writing about? For example, are you sharing marketing tips, doing a product review, covering industry news, or something else?",
                        "options": None,
                        "priority": 1
                    })
                
                if not blog_dict.get("platform"):
                    missing_fields.append({
                        "field": "platform",
                        "question": "Perfect! Where are you planning to publish this? Are you using WordPress, Shopify, Wix, or maybe a custom HTML site?",
                        "options": ["wordpress", "shopify", "wix", "html"],
                        "priority": 2
                    })
                
                if not blog_dict.get("length"):
                    missing_fields.append({
                        "field": "length",
                        "question": "How long are you thinking? Are you going for a quick short read, a medium-length article, or a longer deep dive?",
                        "options": ["short", "medium", "long"],
                        "priority": 3
                    })
                
                if not blog_dict.get("task"):
                    missing_fields.append({
                        "field": "task",
                        "question": "Great! What would you like me to do with this blog post? Should I create it as a draft for you to review, schedule it to publish later, or just save it for now?",
                        "options": ["draft", "schedule", "save"],
                        "priority": 4
                    })
                
                # Sort by priority
                missing_fields.sort(key=lambda x: x.get("priority", 999))
                
                logger.info(f"Missing fields for blog: {missing_fields}")
                if missing_fields:
                    question = self._generate_clarifying_question(missing_fields, "blog")
                    state["response"] = question
                    state["needs_clarification"] = True
                    # Store options for frontend rendering
                    field_info = missing_fields[0]
                    state["options"] = field_info.get("options")
                    logger.info(f"Generated clarifying question: {question}")
                    logger.info(f"Options for frontend: {state['options']}")
                    return state
            
            # If all required fields are present, validate and execute
            # Now we validate the complete payload
            try:
                # Normalize one more time to ensure structure is correct
                normalized_payload = self._normalize_payload(partial_payload.copy())
                logger.info(f"Normalized payload before validation: {json.dumps(normalized_payload, indent=2, default=str)}")
                
                intent_payload = IntentPayload(**normalized_payload)
                payload = intent_payload.content
                
                if not payload:
                    logger.error("Payload is None after validation")
                    state["response"] = "I encountered an error: Content payload is missing. Please try again."
                    state["needs_clarification"] = True
                    return state
                
                logger.info("All required fields present, validating and executing payload")
                result = execute_content_generation(payload, state["user_id"])
            except Exception as validation_error:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Validation error when all fields should be present: {validation_error}")
                logger.error(f"Error traceback: {error_trace}")
                logger.error(f"Partial payload that failed: {json.dumps(partial_payload, indent=2, default=str)}")
                state["response"] = f"I encountered an error validating your request: {str(validation_error)}. Please try again or provide the information in a different way."
                state["needs_clarification"] = True  # Keep asking for clarification
                return state
            
            if result.get("clarifying_question"):
                state["response"] = result["clarifying_question"]
                state["needs_clarification"] = True
            elif result.get("success") and result.get("data"):
                # Store the structured content data in state for frontend
                data = result["data"]
                
                # Ensure images is always a list
                if "images" in data:
                    if not isinstance(data["images"], list):
                        data["images"] = [data["images"]] if data["images"] else []
                    # Filter out any None or empty values
                    data["images"] = [img for img in data["images"] if img and isinstance(img, str) and len(img) > 0]
                
                logger.info(f"📥 Received content generation result from Leo:")
                logger.info(f"  - Data keys: {list(data.keys())}")
                logger.info(f"  - Images: {data.get('images')}")
                logger.info(f"  - Images type: {type(data.get('images'))}")
                logger.info(f"  - Images length: {len(data.get('images', []))}")
                if data.get('images'):
                    for idx, img_url in enumerate(data['images']):
                        logger.info(f"    Image {idx + 1}: {img_url}")
                
                # Get saved_content_id and updated payload from the result
                saved_content_id = data.get("saved_content_id")
                updated_payload = result.get("payload")  # Full payload with content field set
                
                # Store in state
                state["content_data"] = data
                state["response"] = self._format_content_response(data)
                
                # Update partial payload with the full payload returned from Leo (includes content field)
                partial_payload = state.get("partial_payload", {})
                if updated_payload:
                    # Merge the updated payload from Leo into partial_payload
                    # The updated_payload should have the content field set to saved_content_id
                    if "content" not in partial_payload:
                        partial_payload["content"] = {}
                    if "social_media" not in partial_payload.get("content", {}):
                        partial_payload["content"]["social_media"] = {}
                    
                    # Merge the updated payload fields into partial_payload
                    for key, value in updated_payload.items():
                        if value is not None:  # Only update with non-null values
                            partial_payload["content"]["social_media"][key] = value
                    
                    logger.info(f"📝 Merged updated payload from Leo into partial_payload")
                    logger.info(f"   Payload content field: {partial_payload.get('content', {}).get('social_media', {}).get('content')}")
                elif saved_content_id:
                    # Fallback: if payload not returned, manually set content field
                    if "content" not in partial_payload:
                        partial_payload["content"] = {}
                    if "social_media" not in partial_payload.get("content", {}):
                        partial_payload["content"]["social_media"] = {}
                    partial_payload["content"]["social_media"]["content"] = saved_content_id
                    logger.info(f"📝 Fallback: Stored saved_content_id in payload.content.social_media.content: {saved_content_id}")
                
                # Content generation complete - task/date will be handled by posting manager
                # All fields complete, clear partial payload
                state["needs_clarification"] = False
                state["partial_payload"] = None
                
                logger.info(f"✅ Content data stored in state with {len(data.get('images', []))} image(s)")
            elif result.get("error"):
                state["response"] = f"I encountered an error: {result['error']}"
                state["needs_clarification"] = False
            else:
                state["response"] = "I've processed your content generation request."
                state["needs_clarification"] = False
                state["partial_payload"] = None
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error in handle_content_generation: {e}")
            logger.error(f"Error traceback: {error_trace}")
            state["response"] = f"I encountered an error while processing your content generation request: {str(e)}. Please try again."
            state["needs_clarification"] = True  # Keep asking for clarification instead of giving up
        
        return state
    
    def handle_analytics(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Handle analytics intent"""
        try:
            from agents.tools.Orion_Analytics_query import execute_analytics_query
            
            payload = state["intent_payload"].analytics
            if not payload:
                state["response"] = "I need more information about what analytics you'd like to see. Please specify your query."
                return state
            
            result = execute_analytics_query(payload, state["user_id"])
            
            if result.get("clarifying_question"):
                state["response"] = result["clarifying_question"]
            elif result.get("success") and result.get("data"):
                state["response"] = self._format_analytics_response(result["data"])
            elif result.get("error"):
                state["response"] = f"I encountered an error: {result['error']}"
            else:
                state["response"] = "I've processed your analytics query."
                
        except Exception as e:
            logger.error(f"Error in handle_analytics: {e}")
            state["response"] = "I encountered an error while processing your analytics query. Please try again."
        
        return state
    
    def handle_leads_management(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Handle leads management intent"""
        try:
            from agents.tools.Chase_Leads_manager import execute_leads_operation

            # Get user query for direct pattern matching (bypass unreliable LLM)
            user_query = state.get("current_query", "").lower().strip()
            
            # Prefer the unvalidated partial payload (same pattern as content generation)
            partial_payload = state.get("partial_payload", {}) or {}
            leads_dict = partial_payload.get("leads")
            
            # Retrieve asked_questions from state to persist between turns
            asked_questions = state.get("asked_questions", {}) or {}
            
            # Also check in partial_payload for asked_questions
            if partial_payload.get("asked_questions"):
                asked_questions.update(partial_payload.get("asked_questions", {}))

            # PERSISTENCE FIX: Move identified_lead_id to leads_dict for Pydantic payload
            if not leads_dict.get("lead_id") and asked_questions.get("identified_lead_id"):
                leads_dict["lead_id"] = asked_questions["identified_lead_id"]
                logger.info(f"✅ Recovered identified_lead_id: {leads_dict['lead_id']}")

            # CRITICAL FIX: Use direct keyword matching instead of LLM extraction
            # This ensures action is always detected correctly
            if not leads_dict:
                leads_dict = {}
            
            # Detect action from user query using simple pattern matching (no LLM)
            normalized_query = user_query.strip().lower()
            insight_panel_map = {
                "total leads": "summary",
                "leads by status": "status",
                "leads by platform": "platform",
                "time-based trends": "time_trends",
                "time based trends": "time_trends",
                "time trends": "time_trends",
                "weekly trends": "time_trends",
                "weekly comparison": "time_trends"
            }
            if normalized_query in insight_panel_map:
                leads_dict["action"] = "insight"
                leads_dict["insight_type"] = insight_panel_map[normalized_query]
                logger.info(
                    f"✅ Interpreted insight option '{user_query}' as insight_type '{insight_panel_map[normalized_query]}'"
                )
            elif not leads_dict.get("action"):
                if any(keyword in normalized_query for keyword in ["total leads", "insights", "lead stats", "time trends", "weekly", "time-based"]):
                    leads_dict["action"] = "insight"
                    logger.info("✅ Routing generic insight option click to insight action")
                    if "total leads" in normalized_query:
                        leads_dict.setdefault("insight_type", "summary")
                    elif "status" in normalized_query:
                        leads_dict.setdefault("insight_type", "status")
                    elif "platform" in normalized_query:
                        leads_dict.setdefault("insight_type", "platform")
                    elif "time" in normalized_query:
                        leads_dict.setdefault("insight_type", "time_trends")
                if any(keyword in user_query for keyword in ["add lead", "add_lead", "create lead", "new lead", "add a lead"]):
                    leads_dict["action"] = "add_lead"
                    logger.info("✅ Detected action: add_lead from user query")
                elif any(keyword in user_query for keyword in ["update lead", "update_lead", "modify lead", "change lead", "edit lead"]):
                    leads_dict["action"] = "update_lead"
                    logger.info("✅ Detected action: update_lead from user query")
                elif any(keyword in user_query for keyword in ["search lead", "search_lead", "find lead", "look for lead", "search for lead"]):
                    leads_dict["action"] = "search_lead"
                    logger.info("✅ Detected action: search_lead from user query")
                elif any(keyword in user_query for keyword in ["delete lead", "delete_lead", "remove lead", "delete a lead"]):
                    leads_dict["action"] = "delete_lead"
                    logger.info("✅ Detected action: delete_lead from user query")
                elif any(keyword in user_query for keyword in ["insight", "analytics", "stats", "performance"]):
                    leads_dict["action"] = "insight"
                    logger.info("✅ Detected action: insight from user query")
            elif leads_dict.get("action") == "search_lead":
                if any(keyword in user_query for keyword in ["update lead", "update_lead", "modify lead", "change lead", "edit lead"]):
                    leads_dict["action"] = "update_lead"
                    logger.info("✅ Overriding action to update_lead after search results")
                elif any(keyword in user_query for keyword in ["delete lead", "delete_lead", "remove lead", "delete a lead"]):
                    leads_dict["action"] = "delete_lead"
                    logger.info("✅ Overriding action to delete_lead after search results")
            
            # Store leads_dict back to partial_payload
            partial_payload["leads"] = leads_dict
            state["partial_payload"] = partial_payload
            
            # If still no action, ask for clarification
            if not leads_dict or not leads_dict.get("action"):
                state["response"] = "What would you like to do with leads? You can get insights, add a lead, update a lead, search leads, or delete a lead."
                state["needs_clarification"] = True
                state["options"] = ["insight", "add lead", "update lead", "search lead", "delete lead"]
                return state
            
            import re

            def _persist_leads_clarification_state() -> None:
                partial_payload["leads"] = leads_dict
                partial_payload["asked_questions"] = asked_questions
                state["partial_payload"] = partial_payload
                state["asked_questions"] = asked_questions

            if leads_dict.get("action") == "update_lead" and not (leads_dict.get("lead_email") or leads_dict.get("lead_phone")) and asked_questions.get("contact"):
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_query)
                if email_match:
                    leads_dict["lead_email"] = email_match.group(0)
                    logger.info(f"✅ Extracted lead_email (after asking): {leads_dict['lead_email']}")
                    asked_questions.pop("contact", None)
                    _persist_leads_clarification_state()
                else:
                    phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)\.]{6,}[0-9]', user_query)
                    if phone_match:
                        leads_dict["lead_phone"] = phone_match.group(0).strip()
                        logger.info(f"✅ Extracted lead_phone (after asking): {leads_dict['lead_phone']}")
                        asked_questions.pop("contact", None)
                        _persist_leads_clarification_state()

            if leads_dict.get("action") == "delete_lead" and not (leads_dict.get("lead_email") or leads_dict.get("lead_phone")) and asked_questions.get("contact"):
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_query)
                if email_match:
                    leads_dict["lead_email"] = email_match.group(0)
                    logger.info(f"✅ Extracted lead_email for delete disambiguation: {leads_dict['lead_email']}")
                    asked_questions.pop("contact", None)
                    _persist_leads_clarification_state()
                else:
                    phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)\.]{6,}[0-9]', user_query)
                    if phone_match:
                        leads_dict["lead_phone"] = phone_match.group(0).strip()
                        logger.info(f"✅ Extracted lead_phone for delete disambiguation: {leads_dict['lead_phone']}")
                        asked_questions.pop("contact", None)
                        _persist_leads_clarification_state()

            if leads_dict.get("action") == "search_lead" and asked_questions.get("search_criteria"):
                has_answered = False
                if "@" in user_query:
                    leads_dict["lead_email"] = user_query.strip()
                    has_answered = True
                else:
                    phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)\.]{6,}[0-9]', user_query)
                    if phone_match:
                        leads_dict["lead_phone"] = phone_match.group(0).strip()
                        has_answered = True
                    elif user_query and not any(q in user_query.lower() for q in ["what", "how", "when", "where", "which", "why"]):
                        leads_dict["lead_name"] = state.get("current_query", "").strip()
                        has_answered = True

                if has_answered:
                    asked_questions.pop("search_criteria", None)
                    _persist_leads_clarification_state()

            # Extract field values directly from user query for add_lead action (NO LLM!)
            # This handles clarification responses (when we've asked a specific question)
            if leads_dict.get("action") == "add_lead":
                
                # Only do clarification-based extraction here
                # Initial extraction is now done in classify_intent via _extract_lead_data_from_query
                
                # Extract name if we've specifically asked for it
                if not leads_dict.get("lead_name") and asked_questions.get("lead_name"):
                    # If we've asked for name, ANY response is the name (except question words)
                    if not any(q in user_query for q in ["what", "how", "when", "where", "which", "why"]):
                        leads_dict["lead_name"] = state.get("current_query", "").strip()
                        logger.info(f"✅ Extracted lead_name (after asking): {leads_dict['lead_name']}")
                        asked_questions.pop("lead_name", None)  # Mark as answered
                
                # Extract email/phone if we've asked for contact info
                if not (leads_dict.get("lead_email") or leads_dict.get("lead_phone")) and asked_questions.get("contact"):
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_query)
                    if email_match:
                        leads_dict["lead_email"] = email_match.group(0)
                        logger.info(f"✅ Extracted lead_email (after asking): {leads_dict['lead_email']}")
                        asked_questions.pop("contact", None)
                    else:
                        phone_match = re.search(r'[\+\(]?[0-9][0-9\s\-\(\)\.]{6,}[0-9]', user_query)
                        if phone_match:
                            leads_dict["lead_phone"] = phone_match.group(0).strip()
                            logger.info(f"✅ Extracted lead_phone (after asking): {leads_dict['lead_phone']}")
                            asked_questions.pop("contact", None)
                
                # Extract platform if we've specifically asked for it
                if not leads_dict.get("platform") and asked_questions.get("platform"):
                    platforms = ["website", "facebook", "instagram", "linkedin", "referral", "manual", "google", "twitter", "other"]
                    for platform in platforms:
                        if platform in user_query:
                            leads_dict["platform"] = platform
                            logger.info(f"✅ Extracted platform (after asking): {leads_dict['platform']}")
                            asked_questions.pop("platform", None)  # Mark as answered
                            break
                
                # Extract status if we've specifically asked for it
                if not leads_dict.get("status") and asked_questions.get("status"):
                    statuses = ["new", "contacted", "responded", "qualified", "converted", "lost"]
                    for status in statuses:
                        if status in user_query:
                            leads_dict["status"] = status
                            logger.info(f"✅ Extracted status (after asking): {leads_dict['status']}")
                            asked_questions.pop("status", None)  # Mark as answered
                            break
                
                # Extract remarks if we've specifically asked for it
                if not leads_dict.get("remarks") and asked_questions.get("remarks"):
                    # Check if user wants to skip
                    if any(skip_word in user_query for skip_word in ["na", "n/a", "none", "skip"]) or user_query.strip().lower() == "skip":
                        leads_dict["remarks"] = None  # User explicitly skipped
                        logger.info("✅ User skipped remarks (na/skip)")
                        asked_questions.pop("remarks", None)  # Mark as answered
                    elif len(user_query) > 0:
                        # Accept any non-empty response as remarks
                        leads_dict["remarks"] = state.get("current_query", "").strip()
                        logger.info(f"✅ Extracted remarks (after asking): {leads_dict['remarks']}")
                        asked_questions.pop("remarks", None)  # Mark as answered
                
                # CRITICAL: Update both leads_dict AND asked_questions in state
                partial_payload["leads"] = leads_dict
                partial_payload["asked_questions"] = asked_questions
                state["partial_payload"] = partial_payload
                state["asked_questions"] = asked_questions

            # Normalize nested payloads into flattened fields that the tool expects
            # Prioritize insight payload normalization
            leads_dict = leads_dict.copy()
            
            # IMPORTANT: Normalize platform aliases FIRST before other field aliases
            # This ensures platform is set before we check other fields
            platform_aliases = ["source_platform", "source", "platform_source", "source platform"]
            for alias in platform_aliases:
                if leads_dict.get(alias) is not None:
                    # Always normalize platform aliases, even if platform already exists
                    # This ensures consistency
                    if not leads_dict.get("platform") or leads_dict.get("platform") == "":
                        leads_dict["platform"] = leads_dict.pop(alias)
                        logger.info(f"Normalized platform alias '{alias}' to 'platform' in handle_leads_management")
                    else:
                        # If platform already exists, just remove the alias
                        leads_dict.pop(alias, None)
            
            # Normalize field aliases (e.g., "name" -> "lead_name", "phone_number" -> "lead_phone")
            field_aliases = {
                "name": "lead_name",
                "email": "lead_email",
                "phone": "lead_phone",
                "phone_number": "lead_phone",  # Handle phone_number alias
                "id": "lead_id",
            }
            for alias, canonical_field in field_aliases.items():
                if leads_dict.get(alias) is not None and leads_dict.get(canonical_field) is None:
                    leads_dict[canonical_field] = leads_dict.pop(alias)
                # Also handle empty strings - remove if canonical field doesn't exist
                elif leads_dict.get(alias) == "" and leads_dict.get(canonical_field) is None:
                    leads_dict.pop(alias, None)
            
            if leads_dict.get("insight") and isinstance(leads_dict["insight"], dict):
                insight_block = leads_dict["insight"]
                leads_dict.setdefault("lead_id", insight_block.get("lead_id"))
                leads_dict.setdefault("lead_name", insight_block.get("lead_name"))
                leads_dict.setdefault("platform", insight_block.get("platform"))
                leads_dict.setdefault("date_range", insight_block.get("date_range"))
                leads_dict.setdefault("insight_type", insight_block.get("insight_type"))
                # Set action to insight if not already set
                if not leads_dict.get("action"):
                    leads_dict["action"] = "insight"
            
            if leads_dict.get("add_lead") and isinstance(leads_dict["add_lead"], dict):
                add_block = leads_dict["add_lead"]
                
                # IMPORTANT: Normalize platform aliases FIRST in nested block
                platform_aliases = ["source_platform", "source", "platform_source", "source platform"]
                for alias in platform_aliases:
                    if add_block.get(alias) is not None:
                        if not add_block.get("platform") or add_block.get("platform") == "":
                            add_block["platform"] = add_block.pop(alias)
                            logger.info(f"Normalized platform alias '{alias}' to 'platform' in add_block")
                        else:
                            # If platform already exists, just remove the alias
                            add_block.pop(alias, None)
                
                # Normalize field aliases in nested block
                field_aliases = {
                    "name": "lead_name",
                    "email": "lead_email",
                    "phone": "lead_phone",
                    "phone_number": "lead_phone",  # Handle phone_number alias
                    "id": "lead_id",
                }
                for alias, canonical_field in field_aliases.items():
                    if add_block.get(alias) is not None and add_block.get(canonical_field) is None:
                        add_block[canonical_field] = add_block.pop(alias)
                    elif add_block.get(alias) == "" and add_block.get(canonical_field) is None:
                        add_block.pop(alias, None)
                
                # Always copy values from nested block to top level
                # IMPORTANT: Use explicit assignment to ensure values are copied, even if they exist
                # This ensures that new values from nested blocks overwrite existing ones
                if "lead_name" in add_block and add_block["lead_name"] is not None:
                    leads_dict["lead_name"] = add_block["lead_name"]
                if "lead_email" in add_block and add_block["lead_email"] is not None:
                    leads_dict["lead_email"] = add_block["lead_email"]
                if "lead_phone" in add_block and add_block["lead_phone"] is not None:
                    leads_dict["lead_phone"] = add_block["lead_phone"]
                if "platform" in add_block and add_block["platform"] is not None:
                    # Always update platform if it exists in nested block
                    leads_dict["platform"] = add_block["platform"]
                    logger.info(f"Copied platform '{add_block['platform']}' from add_block to leads_dict")
                if "status" in add_block and add_block["status"] is not None:
                    leads_dict["status"] = add_block["status"]
                if "lead_id" in add_block and add_block["lead_id"] is not None:
                    leads_dict["lead_id"] = add_block["lead_id"]
                if "remarks" in add_block:
                    remarks_value = add_block["remarks"]
                    # Handle "na" for optional remarks - convert to None
                    if remarks_value and isinstance(remarks_value, str) and remarks_value.lower().strip() == "na":
                        leads_dict["remarks"] = None
                    elif remarks_value is not None:
                        leads_dict["remarks"] = remarks_value
                if "follow_up" in add_block and add_block["follow_up"] is not None:
                    leads_dict["follow_up"] = add_block["follow_up"]
                if not leads_dict.get("action"):
                    leads_dict["action"] = "add_lead"
            
            # Also check for email/phone at top level of leads_dict (not in nested blocks)
            # This handles cases where LLM puts them directly in leads dict
            # Use the outer scope field_aliases
            top_level_aliases = {
                "name": "lead_name",
                "email": "lead_email",
                "phone": "lead_phone",
                "phone_number": "lead_phone",  # Handle phone_number alias
                "id": "lead_id",
            }
            for alias, canonical_field in top_level_aliases.items():
                if leads_dict.get(alias) is not None and not leads_dict.get(canonical_field):
                    leads_dict[canonical_field] = leads_dict.pop(alias)
            
            # Also normalize platform aliases at top level
            platform_aliases = ["source_platform", "source", "platform_source", "source platform"]
            for alias in platform_aliases:
                if leads_dict.get(alias) is not None and not leads_dict.get("platform"):
                    leads_dict["platform"] = leads_dict.pop(alias)
                    logger.info(f"Normalized platform alias '{alias}' to 'platform' at top level")

            if leads_dict.get("update_lead") and isinstance(leads_dict["update_lead"], dict):
                upd_block = leads_dict["update_lead"]
                leads_dict.setdefault("lead_name", upd_block.get("lead_name"))
                leads_dict.setdefault("lead_email", upd_block.get("lead_email"))
                leads_dict.setdefault("lead_phone", upd_block.get("lead_phone"))
                leads_dict.setdefault("lead_id", upd_block.get("lead_id"))
                leads_dict.setdefault("remarks", upd_block.get("remarks"))
                leads_dict.setdefault("follow_up", upd_block.get("follow_up"))

            if leads_dict.get("search_lead") and isinstance(leads_dict["search_lead"], dict):
                sea_block = leads_dict["search_lead"]
                leads_dict.setdefault("key", sea_block.get("key"))
                leads_dict.setdefault("lead_id", sea_block.get("lead_id"))
                leads_dict.setdefault("date_range", sea_block.get("date_range") or leads_dict.get("date_range"))

            # Note: "na" for remarks is kept as-is during validation
            # It will be normalized to None when executing the operation

            try:
                payload = LeadsManagementPayload(**leads_dict)
            except Exception as e:
                logger.error(f"Failed to validate leads payload: {e}")
                state["response"] = "I couldn't understand the leads request. Please specify the action (insights, add, update, search, or delete)."
                state["needs_clarification"] = True
                state["options"] = ["insight", "add lead", "update lead", "search lead", "delete lead"]
                return state

            # IMPORTANT: Update partial_payload with normalized leads_dict before checking missing fields
            # This ensures the normalized values are persisted
            partial_payload["leads"] = leads_dict
            state["partial_payload"] = partial_payload
          
            # Ask for missing required fields with clear priority
            missing_fields = self._get_missing_fields_for_leads(payload, asked_questions)

            if missing_fields:
                field_info = missing_fields[0]

                state["response"] = field_info["question"]
                state["needs_clarification"] = True
                state["options"] = field_info.get("options")
                state["last_clarification_field"] = field_info["field"]
                
                # CRITICAL: Store clarification flags in partial_payload so they persist in cache
                partial_payload["_needs_clarification"] = True
                partial_payload["_last_clarification_field"] = field_info["field"]
                state["partial_payload"] = partial_payload
                
                logger.info(f"❓ Asking clarification for field: '{field_info['field']}' (stored in cache)")

                return state

            payload_action = payload.action
            result = execute_leads_operation(payload, state["user_id"], asked_questions)

            if result.get("clarifying_question"):
                state["response"] = result["clarifying_question"]
                state["needs_clarification"] = True
                state["options"] = result.get("options")
                
                # Get the field name being asked from result if possible
                # The tool returns key in asked_questions, we can extract it
                asked = result.get("asked_questions", {})
                last_key = list(asked.keys())[-1] if asked else None
                
                if last_key:
                    state["last_clarification_field"] = last_key
                    if "partial_payload" not in state or state["partial_payload"] is None:
                        state["partial_payload"] = {}
                    state["partial_payload"]["_needs_clarification"] = True
                    state["partial_payload"]["_last_clarification_field"] = last_key
                
                if result.get("asked_questions"):
                    state["asked_questions"] = result["asked_questions"]
                    if "partial_payload" not in state:
                        state["partial_payload"] = {}
                    state["partial_payload"]["asked_questions"] = result["asked_questions"]
                state["partial_payload"] = partial_payload
                return state
            elif result.get("success") and result.get("data"):
                data = result["data"]

                if payload_action == "search_lead":
                    leads_list = data.get("leads") or []
                    lead_for_context = data.get("selected_lead") or (leads_list[0] if leads_list else None)

                    response_lines = ["Here are the lead details I found:"]
                    if lead_for_context:
                        response_lines.extend(self._build_lead_info_lines(lead_for_context))
                    else:
                        response_lines.append("• Unable to fetch detailed lead information right now.")

                    if lead_for_context and len(leads_list) > 1:
                        response_lines.append(f"({len(leads_list)} total matches found; showing the most recent one.)")

                    response_lines.append("")
                    response_lines.append("Would you like to update this lead or delete it?")

                    state["response"] = "\n".join(response_lines)
                    state["needs_clarification"] = True
                    state["options"] = ["update lead", "delete  lead"]

                    if lead_for_context:
                        leads_ctx = partial_payload.setdefault("leads", {})

                        field_mappings = {
                            "lead_id": ["lead_id", "id"],
                            "lead_name": ["lead_name", "name"],
                        }
                        for canonical, keys in field_mappings.items():
                            for key in keys:
                                value = lead_for_context.get(key)
                                if value:
                                    leads_ctx[canonical] = value
                                    break

                        identified_id = lead_for_context.get("lead_id") or lead_for_context.get("id")
                        if identified_id:
                            asked_questions["identified_lead_id"] = identified_id
                        identified_name = lead_for_context.get("lead_name") or lead_for_context.get("name")
                        if identified_name:
                            asked_questions["identified_lead_name"] = identified_name

                        partial_payload["leads"] = leads_ctx

                    state["partial_payload"] = partial_payload
                    state["asked_questions"] = asked_questions
                    if result.get("content_data"):
                        state["content_data"] = result.get("content_data")
                    return state

                state["response"] = self._format_leads_response(data)
                if payload_action == "insight":
                    insight_context = data.get("insight_context", {}) or {}
                    leads_context = partial_payload.setdefault("leads", {})
                    for key, value in insight_context.items():
                        if value:
                            leads_context[key] = value
                    partial_payload["leads"] = leads_context
                    state["partial_payload"] = partial_payload
                    state["asked_questions"] = asked_questions
                    leads_dict = leads_context
                    insight_options = data.get("insights", {}).get("options")
                    state["options"] = insight_options or [
                        "Total leads",
                        "Leads by status",
                        "Leads by platform",
                        "Time-based trends"
                    ]
                else:
                    state["options"] = result.get("options")
                    state.pop("partial_payload", None)
                    state.pop("asked_questions", None)
                state["needs_clarification"] = False
                if result.get("content_data"):
                    state["content_data"] = result.get("content_data")
            elif result.get("error"):
                state["response"] = f"I encountered an error: {result['error']}"
            else:
                state["response"] = "I've processed your leads management request."
                
        except Exception as e:
            logger.error(f"Error in handle_leads_management: {e}")
            state["response"] = "I encountered an error while processing your leads request. Please try again."
        
        return state
    
    def handle_posting_manager(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Handle posting manager intent"""
        try:
            from agents.tools.Emily_post_manager import execute_posting_operation
            
            payload = state["intent_payload"].posting
            if not payload:
                state["response"] = "I need more information about what you'd like to do with your posts. Please specify the action."
                return state
            
            result = execute_posting_operation(payload, state["user_id"])
            
            if result.get("clarifying_question"):
                state["response"] = result["clarifying_question"]
            elif result.get("success") and result.get("data"):
                state["response"] = self._format_posting_response(result["data"])
            elif result.get("error"):
                state["response"] = f"I encountered an error: {result['error']}"
            else:
                state["response"] = "I've processed your posting request."
                
        except Exception as e:
            logger.error(f"Error in handle_posting_manager: {e}")
            state["response"] = "I encountered an error while processing your posting request. Please try again."
        
        return state
    
    def handle_general_talks(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Handle general conversational intent"""
        try:
            from agents.tools.general_chat_tool import execute_general_chat
            
            payload = state["intent_payload"].general
            if not payload:
                payload = GeneralTalkPayload(message=state["current_query"])
            
            result = execute_general_chat(payload, state["user_id"])
            
            if result.get("success") and result.get("data"):
                state["response"] = result["data"]
            elif result.get("error"):
                state["response"] = f"I encountered an error: {result['error']}"
            else:
                state["response"] = "I'm here to help! How can I assist you today?"
                
        except Exception as e:
            logger.error(f"Error in handle_general_talks: {e}")
            state["response"] = "I'm here to help! How can I assist you today?"
        
        return state
    
    def generate_final_response(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Final response formatting node"""
        # Response is already set in handler nodes
        # This node can be used for additional formatting if needed
        return state
    
    def _format_content_response(self, data: Any) -> str:
        """Format content generation response with structured content"""
        if isinstance(data, dict):
            # Check if this is structured content (title, content, hashtags, images)
            # If it has structured content data, return a simple message (card will be displayed separately)
            if "title" in data and "content" in data:
                # Return simple message - the card will be displayed from content_data
                return "Here is the post you requested"
            elif "message" in data:
                return data["message"]
            elif "content" in data:
                return data["content"]
        return str(data)
    
    def _format_analytics_response(self, data: Any) -> str:
        """Format analytics response"""
        if isinstance(data, dict):
            if "message" in data:
                return data["message"]
            elif "summary" in data:
                return data["summary"]
        return str(data)
    
    def _format_leads_response(self, data: Any) -> str:
        """Format leads management response"""
        if isinstance(data, dict):
            if "message" in data:
                return data["message"]
            elif "summary" in data:
                return data["summary"]
        return str(data)

    def _build_lead_info_lines(self, lead: Dict[str, Any]) -> List[str]:
        """Helper to build a clean, bullet-style summary for a single lead."""
        def pick(*keys: str) -> Optional[str]:
            for key in keys:
                value = lead.get(key)
                if value:
                    return value
            return None

        lines: List[str] = []
        name = pick("lead_name", "name")
        if name:
            lines.append(f"• Name: {name}")

        status = pick("status", "lead_status")
        if status:
            lines.append(f"• Initial status: {status}")
        else:
            lines.append(f"• Initial status: Not specified")

        platform = pick("platform", "source_platform")
        if platform:
            lines.append(f"• Platform: {platform}")

        email = pick("lead_email", "email")
        if email:
            lines.append(f"• Email: {email}")

        phone = pick("lead_phone", "phone_number")
        if phone:
            lines.append(f"• Phone: {phone}")

        return lines
    
    def _format_posting_response(self, data: Any) -> str:
        """Format posting manager response"""
        if isinstance(data, dict):
            if "message" in data:
                return data["message"]
            elif "summary" in data:
                return data["summary"]
        return str(data)
    
    def chat(self, user_id: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Main chat interface (deprecated - use get_intent_based_response instead)"""
        try:
            # This method is kept for backward compatibility
            # The actual implementation is now in get_intent_based_response
            initial_state: IntentBasedChatbotState = {
                "user_id": user_id,
                "current_query": query,
                "conversation_history": conversation_history or [],
                "intent_payload": None,
                "partial_payload": None,
                "response": None,
                "context": {},
                "needs_clarification": False,
                "options": None,
                "content_data": None
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return result.get("response", "I apologize, but I couldn't process your request.")
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I encountered an error while processing your request."

# Global instance
_intent_based_chatbot = None

# In-memory cache for partial payloads (keyed by user_id)
# In production, this could be stored in Redis or database
_partial_payload_cache: Dict[str, Dict[str, Any]] = {}

def clear_partial_payload_cache(user_id: str) -> None:
    """Clear the partial payload cache for a specific user"""
    _partial_payload_cache.pop(user_id, None)
    logger.info(f"Cleared partial payload cache for user {user_id}")

def get_intent_based_chatbot() -> IntentBasedChatbot:
    """Get or create the intent-based chatbot instance"""
    global _intent_based_chatbot
    if _intent_based_chatbot is None:
        _intent_based_chatbot = IntentBasedChatbot()
    return _intent_based_chatbot

def get_intent_based_response(user_id: str, message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Get response from intent-based chatbot"""
    chatbot = get_intent_based_chatbot()
    
    # Retrieve partial payload from cache if exists
    partial_payload = _partial_payload_cache.get(user_id)
    
    # Create state with partial payload
    initial_state: IntentBasedChatbotState = {
        "user_id": user_id,
        "current_query": message,
        "conversation_history": conversation_history or [],
        "intent_payload": None,
        "partial_payload": partial_payload,
        "response": None,
        "context": {},
        "needs_clarification": False,
        "options": None,
        "content_data": None
    }
    
    # Run the graph
    result = chatbot.graph.invoke(initial_state)
    
    # Update cache with new partial payload if clarification is needed
    if result.get("needs_clarification") and result.get("partial_payload"):
        _partial_payload_cache[user_id] = result["partial_payload"]
    elif not result.get("needs_clarification"):
        # Clear cache when request is complete
        _partial_payload_cache.pop(user_id, None)
    
    return {
        "response": result.get("response", "I apologize, but I couldn't process your request."),
        "options": result.get("options"),
        "content_data": result.get("content_data")  # Include structured content data (title, content, hashtags, images)
    }

def get_intent_based_response_stream(user_id: str, message: str, conversation_history: Optional[List[Dict[str, str]]] = None):
    """Stream response from intent-based chatbot"""
    chatbot = get_intent_based_chatbot()
    
    # Retrieve partial payload from cache if exists
    partial_payload = _partial_payload_cache.get(user_id)
    
    # Create state with partial payload
    initial_state: IntentBasedChatbotState = {
        "user_id": user_id,
        "current_query": message,
        "conversation_history": conversation_history or [],
        "intent_payload": None,
        "partial_payload": partial_payload,
        "response": None,
        "context": {},
        "needs_clarification": False,
        "options": None,
        "content_data": None
    }
    
    # Run the graph
    result = chatbot.graph.invoke(initial_state)
    
    # Update cache with new partial payload if clarification is needed
    if result.get("needs_clarification") and result.get("partial_payload"):
        _partial_payload_cache[user_id] = result["partial_payload"]
    elif not result.get("needs_clarification"):
        # Clear cache when request is complete
        _partial_payload_cache.pop(user_id, None)
    
    response = result.get("response", "I apologize, but I couldn't process your request.")
    options = result.get("options")
    content_data = result.get("content_data")
    
    # Debug: Log what's in the result
    logger.info(f"🔍 Stream function - Result keys: {list(result.keys())}")
    logger.info(f"   Result has 'content_data' key: {'content_data' in result}")
    logger.info(f"   content_data value: {content_data}")
    logger.info(f"   content_data type: {type(content_data)}")
    if content_data:
        logger.info(f"   content_data keys: {list(content_data.keys()) if isinstance(content_data, dict) else 'N/A'}")
        logger.info(f"   content_data images: {content_data.get('images') if isinstance(content_data, dict) else 'N/A'}")
    
    # For now, yield the full response in chunks
    # Can be enhanced later for true streaming
    chunk_size = 10
    for i in range(0, len(response), chunk_size):
        yield response[i:i + chunk_size]
    
    # Yield options at the end if they exist
    if options:
        yield f"\n\nOPTIONS:{json.dumps(options)}"
    
    # Yield content_data at the end if it exists
    if content_data:
        logger.info(f"📤 Yielding CONTENT_DATA in stream: {json.dumps(content_data, default=str)[:200]}...")
        logger.info(f"   Images in content_data: {content_data.get('images')}")
        try:
            content_data_json = json.dumps(content_data, default=str)
            yield f"\n\nCONTENT_DATA:{content_data_json}"
            logger.info(f"✅ Successfully yielded CONTENT_DATA")
        except Exception as e:
            logger.error(f"❌ Error serializing content_data: {e}", exc_info=True)
    else:
        logger.warning(f"⚠️ content_data is None or empty - not yielding CONTENT_DATA")
        logger.warning(f"   Result keys: {list(result.keys())}")
        logger.warning(f"   Checking result directly: {result.get('content_data', 'NOT_FOUND')}")


