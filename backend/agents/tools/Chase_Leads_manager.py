"""
Chase Leads Manager Tool
Handles all lead management operations (add, update, search, delete, insight)

- add_lead: Add a new lead to the system
- update_lead: Update an existing lead's information
- search_lead: Search for leads by name, email, phone, or keywords
- delete_lead: Delete a lead by ID (or by name with disambiguation) after confirmation
- insight: Generate insights and analytics for a specific lead

Each operation validates required fields and returns either:
- Success with data message
- Clarifying question if required fields are missing
- Error message if operation fails
"""

import logging
import re
from typing import Dict, Any, Optional
from agents.emily import LeadsManagementPayload

logger = logging.getLogger(__name__)


def validate_email(email: Optional[str]) -> tuple:
    """
    Validate email address format without using LLM
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if email is valid, False otherwise
        - error_message: Error message if invalid, None if valid
    """
    if not email:
        return True, None  # Empty email is allowed (optional field)
    
    email = email.strip()
    
    # Basic email regex pattern
    # Pattern: local@domain
    # Local part: alphanumeric, dots, hyphens, underscores, plus signs
    # Domain part: alphanumeric, dots, hyphens
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False, f"Invalid email format: '{email}'. Please provide a valid email address (e.g., name@example.com)."
    
    # Additional checks
    if email.startswith('.') or email.startswith('@'):
        return False, f"Invalid email format: '{email}'. Email cannot start with a dot or @ symbol."
    
    if '..' in email or email.count('@') != 1:
        return False, f"Invalid email format: '{email}'. Email must contain exactly one @ symbol and no consecutive dots."
    
    # Check domain has at least one dot (TLD)
    parts = email.split('@')
    if len(parts) != 2:
        return False, f"Invalid email format: '{email}'. Email must have format: name@domain.com"
    
    domain = parts[1]
    if '.' not in domain or domain.startswith('.') or domain.endswith('.'):
        return False, f"Invalid email format: '{email}'. Domain must be valid (e.g., example.com)."
    
    return True, None


def validate_phone(phone: Optional[str]) -> tuple:
    """
    Validate phone number format without using LLM
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if phone is valid, False otherwise
        - error_message: Error message if invalid, None if valid
    """
    if not phone:
        return True, None  # Empty phone is allowed (optional field)
    
    phone = phone.strip()
    
    # Remove common formatting characters for validation
    cleaned_phone = re.sub(r'[\s\-\(\)\.]', '', phone)
    
    # Check if phone starts with + (international format)
    if cleaned_phone.startswith('+'):
        cleaned_phone = cleaned_phone[1:]
    
    # Phone should contain only digits after removing formatting
    if not cleaned_phone.isdigit():
        return False, f"Invalid phone number format: '{phone}'. Phone number should contain only digits, spaces, dashes, parentheses, dots, and optionally a + prefix (e.g., +1234567890 or (123) 456-7890)."
    
    # Check minimum length (at least 7 digits, typically 10-15 digits)
    if len(cleaned_phone) < 7:
        return False, f"Invalid phone number format: '{phone}'. Phone number is too short (minimum 7 digits required)."
    
    # Check maximum length (typically 15 digits for international format)
    if len(cleaned_phone) > 15:
        return False, f"Invalid phone number format: '{phone}'. Phone number is too long (maximum 15 digits allowed)."
    
    return True, None

def execute_leads_operation(payload: LeadsManagementPayload, user_id: str, asked_questions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute leads operation based on the payload
    
    Routing Logic (Prioritized):
    - If insight payload or action exists → prioritize and route to _handle_insight()
    - If no action specified → return clarifying question
    - Route to handler based on payload.action:
      * "insight" → _handle_insight() (prioritized)
      * "add_lead" → _handle_add_lead()
      * "update_lead" → _handle_update_lead()
      * "search_lead" → _handle_search_lead()
      * "delete_lead" → _handle_delete_lead()
      * Unknown action → return error
    
    Args:
        payload: LeadsManagementPayload with operation details
        user_id: User ID for the request
        asked_questions: Dict tracking which questions have been asked (key: field_name, value: question_text)
        
    Returns:
        Dict with one of:
        - {"success": False, "clarifying_question": "..."} if missing action
        - {"success": False, "error": "..."} if unknown action or error
        - {"success": True, "data": {...}} if operation succeeds
        - {"success": False, "clarifying_question": "..."} from handler if missing required fields
    """
    try:
        # Initialize asked_questions if not provided
        if asked_questions is None:
            asked_questions = {}
        
        # Prioritize insight action - check for insight payload or action first
        if payload.insight or payload.action == "insight":
            return _handle_insight(payload, user_id, asked_questions)
        
        # If no action specified, ask for clarification
        if not payload.action:
            return {
                "success": False,
                "clarifying_question": "What would you like to do with leads? (get insights, add a lead, update a lead, search leads, delete a lead)"
            }
        
        # Route to appropriate handler
        if payload.action == "add_lead":
            return _handle_add_lead(payload, user_id, asked_questions)
        elif payload.action == "update_lead":
            return _handle_update_lead(payload, user_id, asked_questions)
        elif payload.action == "search_lead":
            return _handle_search_lead(payload, user_id, asked_questions)
        elif payload.action == "delete_lead":
            return _handle_delete_lead(payload, user_id, asked_questions)
        else:
            return {
                "success": False,
                "error": f"Unknown action: {payload.action}"
            }
            
    except Exception as e:
        logger.error(f"Error in execute_leads_operation: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def _handle_add_lead(payload: LeadsManagementPayload, user_id: str, asked_questions: Dict[str, Any]) -> Dict[str, Any]:
    """Handle adding a new lead with email and phone validation"""
    # Remove fields from asked_questions when they are provided (user answered)
    if payload.lead_name:
        asked_questions.pop("lead_name", None)
    
    # Check for lead_name
    if not payload.lead_name:
        question_key = "lead_name"
        question_text = "What is the lead's name?"
        
        # Check if we've asked this question before
        if question_key in asked_questions:
            return {
                "success": False,
                "clarifying_question": f"I already asked for the lead's name earlier. Please provide the lead's name to continue. Previously asked: '{asked_questions[question_key]}'",
                "asked_questions": asked_questions
            }
        
        asked_questions[question_key] = question_text
        return {
            "success": False,
            "clarifying_question": question_text,
            "asked_questions": asked_questions
        }
    
    # Remove email from asked_questions if provided (user answered)
    if payload.lead_email:
        asked_questions.pop("lead_email_validation", None)
        asked_questions.pop("lead_email", None)
        asked_questions.pop("contact", None)  # Also remove contact question if email provided
    
    # Validate email if provided
    if payload.lead_email:
        is_valid, error_msg = validate_email(payload.lead_email)
        if not is_valid:
            question_key = "lead_email_validation"
            if question_key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": f"I already mentioned the email validation issue. {error_msg} Please provide a valid email address or leave it empty if you don't have one.",
                    "asked_questions": asked_questions
                }
            asked_questions[question_key] = error_msg
            return {
                "success": False,
                "clarifying_question": error_msg,
                "asked_questions": asked_questions
            }
    
    # Remove phone from asked_questions if provided (user answered)
    if payload.lead_phone:
        asked_questions.pop("lead_phone_validation", None)
        asked_questions.pop("lead_phone", None)
        asked_questions.pop("contact", None)  # Also remove contact question if phone provided
    
    # Validate phone if provided
    if payload.lead_phone:
        is_valid, error_msg = validate_phone(payload.lead_phone)
        if not is_valid:
            question_key = "lead_phone_validation"
            if question_key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": f"I already mentioned the phone number validation issue. {error_msg} Please provide a valid phone number or leave it empty if you don't have one.",
                    "asked_questions": asked_questions
                }
            asked_questions[question_key] = error_msg
            return {
                "success": False,
                "clarifying_question": error_msg,
                "asked_questions": asked_questions
            }
    
    # Check that at least email or phone is provided
    if not payload.lead_email:
        question_key = "lead_email"
        question_text = "What is the lead's email address?"
        
        if question_key in asked_questions:
            return {
                "success": False,
                "clarifying_question": f"I already asked for the lead's email. Please provide an email address to continue. Previously asked: '{asked_questions[question_key]}'",
                "asked_questions": asked_questions
            }
        
        asked_questions[question_key] = question_text
        return {
            "success": False,
            "clarifying_question": question_text,
            "asked_questions": asked_questions
        }

    if not payload.lead_phone:
        question_key = "lead_phone"
        question_text = "Can you share the lead's phone number (include the country code if possible)?"
        
        if question_key in asked_questions:
            return {
                "success": False,
                "clarifying_question": f"I already asked for the lead's phone number. Please provide it to continue. Previously asked: '{asked_questions[question_key]}'",
                "asked_questions": asked_questions
            }
        
        asked_questions[question_key] = question_text
        return {
            "success": False,
            "clarifying_question": question_text,
            "asked_questions": asked_questions
        }
    
    # Remove platform from asked_questions if provided
    if payload.platform:
        asked_questions.pop("platform", None)
    
    # Check for platform (optional but recommended)
    if not payload.platform:
        question_key = "platform"
        question_text = "What is the source platform for this lead?"
        
        if question_key not in asked_questions:
            asked_questions[question_key] = question_text
            return {
                "success": False,
                "clarifying_question": question_text,
                "options": ["website", "facebook", "instagram", "linkedin", "referral", "google", "twitter", "other"],
                "asked_questions": asked_questions
            }
    
    # Remove status from asked_questions if provided
    if payload.status:
        asked_questions.pop("status", None)
    
    # Check for status (optional but recommended)
    if not payload.status:
        question_key = "status"
        question_text = "What is the initial status of this lead?"
        
        if question_key not in asked_questions:
            asked_questions[question_key] = question_text
            return {
                "success": False,
                "clarifying_question": question_text,
                "options": ["new", "contacted", "responded", "qualified", "converted", "lost"],
                "asked_questions": asked_questions
            }
    
    # Check if we already asked for remarks
    if "remarks" in asked_questions:
        # We asked, so user must have answered - pop it regardless of value
        asked_questions.pop("remarks", None)
        logger.info("✅ Remarks question answered (marked as complete)")
    # Check for remarks (optional) - only ask if not already asked
    elif not payload.remarks:
        question_key = "remarks"
        question_text = "Any additional information about this lead? (Enter 'na' if none)"
        asked_questions[question_key] = question_text
        return {
            "success": False,
            "clarifying_question": question_text,
            "asked_questions": asked_questions
        }
    
    # ACTUAL SUPABASE STORAGE IMPLEMENTATION
    try:
        from supabase import create_client
        import os
        from datetime import datetime
        
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found")
            return {
                "success": False,
                "error": "Database configuration error. Please contact support."
            }
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Normalize remarks (convert "na" to None)
        final_remarks = None
        if payload.remarks:
            remarks_lower = str(payload.remarks).lower().strip()
            if remarks_lower not in ["na", "n/a", "none"]:
                final_remarks = payload.remarks
        
        # Prepare lead data for insertion (match existing database schema)
        lead_data = {
            "user_id": user_id,
            "name": payload.lead_name,  # Database uses 'name' not 'lead_name'
            "email": payload.lead_email,  # Database uses 'email' not 'lead_email'
            "phone_number": payload.lead_phone,  # Database uses 'phone_number' not 'lead_phone'
            "source_platform": payload.platform or "other",  # Database uses 'source_platform' not 'platform'
            "status": payload.status or "new",
            "form_data": {},
            "metadata": {
                "created_via_chatbot": True,
                "created_at": datetime.utcnow().isoformat(),
                "remarks": final_remarks
            }
        }
        
        logger.info(f"📝 Attempting to save lead to Supabase: {lead_data}")
        
        # Insert into Supabase leads table
        result = supabase.table("leads").insert(lead_data).execute()
        
        if result.data:
            lead_id = result.data[0].get("id") if result.data else None
            logger.info(f"✅ Successfully created lead: {payload.lead_name} (ID: {lead_id})")
            
            # Build success message
            message_parts = [
                f"✅ Successfully added {payload.lead_name} as a new lead!",
                "",
                "📋 Details:",
                f"• Name: {payload.lead_name}"
            ]
            
            if payload.lead_email:
                message_parts.append(f"• Email: {payload.lead_email}")
            if payload.lead_phone:
                message_parts.append(f"• Phone: {payload.lead_phone}")
            
            message_parts.append(f"• Platform: {payload.platform or 'other'}")
            message_parts.append(f"• Status: {payload.status or 'new'}")
            
            if final_remarks:
                message_parts.append(f"• Notes: {final_remarks}")
            
            message_parts.append("")
            message_parts.append("The lead is now visible on your leads page!")
            
            return {
                "success": True,
                "data": {
                    "message": "\n".join(message_parts),
                },
                "asked_questions": {}  # Clear asked_questions after success
            }
        else:
            logger.error(f"Failed to create lead: No data returned from Supabase")
            return {
                "success": False,
                "error": "Failed to save lead to database. Please try again."
            }
            
    except Exception as e:
        logger.error(f"❌ Error saving lead to Supabase: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }

def _handle_update_lead(payload: LeadsManagementPayload, user_id: str, asked_questions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update lead flow:

    1) Ask for lead NAME first (if no id).
    2) Check if name exists for this user:
       - 0 matches  -> tell user lead doesn't exist.
       - >1 matches -> ask for phone/email to verify.
    3) Once a single lead is identified:
       - If no update fields yet -> ask what to update (name, email, phone, platform, status, remarks, follow_up).
       - If fields present -> validate email/phone and apply update in Supabase.
    """

    # --- Normalize possible nested update_lead payload ---
    upd = payload.update_lead
    lead_id = payload.lead_id or asked_questions.get("identified_lead_id")
    lead_name = payload.lead_name or (upd.lead_name if upd else None)
    lead_email = payload.lead_email or (upd.lead_email if upd else None)
    lead_phone = payload.lead_phone or (upd.lead_phone if upd else None)
    platform = payload.platform or (upd.platform if upd else None)
    remarks = payload.remarks or (upd.remarks if upd else None)
    follow_up = payload.follow_up or (upd.follow_up if upd else None)
    status = payload.status

    # --- 1) Ask for NAME first if no identifier ---
    if not lead_id and not lead_name:
        key = "update_lead_name"
        q = "What is the name of the lead you want to update?"
        if key in asked_questions:
            return {
                "success": False,
                "clarifying_question": (
                    "I still need the lead's name to update it. "
                    f"Please tell me the exact lead name. Previously I asked: '{asked_questions[key]}'"
                ),
                "asked_questions": asked_questions,
            }
        asked_questions[key] = q
        return {
            "success": False,
            "clarifying_question": q,
            "asked_questions": asked_questions,
        }

    # --- 2) Validate update email/phone if provided ---
    if lead_email:
        ok, err = validate_email(lead_email)
        if not ok:
            key = "lead_email_validation"
            if key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": (
                        f"I already mentioned an email issue. {err} "
                        "Please send a valid email or remove it if you don't want to update it."
                    ),
                    "asked_questions": asked_questions,
                }
            asked_questions[key] = err
            return {
                "success": False,
                "clarifying_question": err,
                "asked_questions": asked_questions,
            }

    if lead_phone:
        ok, err = validate_phone(lead_phone)
        if not ok:
            key = "lead_phone_validation"
            if key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": (
                        f"I already mentioned a phone issue. {err} "
                        "Please send a valid phone number or remove it if you don't want to update it."
                    ),
                    "asked_questions": asked_questions,
                }
            asked_questions[key] = err
            return {
                "success": False,
                "clarifying_question": err,
                "asked_questions": asked_questions,
            }

    # --- 3) Connect to Supabase and resolve lead ---
    try:
        from supabase import create_client
        import os, json

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials missing in _handle_update_lead")
            return {
                "success": False,
                "error": "Database configuration error. Please contact support.",
            }

        supabase = create_client(supabase_url, supabase_key)

        lead_row = None

        if lead_id:
            res = (
                supabase.table("leads")
                .select("id,name,email,phone_number,source_platform,status,metadata")
                .eq("id", lead_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if res.data:
                lead_row = res.data[0]
                asked_questions["identified_lead_id"] = lead_row["id"] # Persist ID for next turn
        else:
            # Name-based lookup for this user
            q = (
                supabase.table("leads")
                .select("id,name,email,phone_number,source_platform,status,metadata")
                .eq("user_id", user_id)
                .ilike("name", f"%{lead_name}%")
            )
            if lead_email:
                q = q.ilike("email", f"%{lead_email}%")
            if lead_phone:
                q = q.ilike("phone_number", f"%{lead_phone}%")

            res = q.limit(10).execute()
            rows = res.data or []

            if len(rows) == 0:
                return {
                    "success": False,
                    "error": f"No lead found with the name '{lead_name}'.",
                }
            if len(rows) == 1:
                lead_row = rows[0]
                asked_questions["identified_lead_id"] = lead_row["id"] # Persist ID for next turn
            else:
                if not lead_email and not lead_phone:
                    options = []
                    for r in rows:
                        parts = [r.get("name") or ""]
                        if r.get("email"):
                            parts.append(f"email: {r.get('email')}")
                        if r.get("phone_number"):
                            parts.append(f"phone: {r.get('phone_number')}")
                        options.append(" | ".join(parts))

                    key = "contact"
                    q_text = (
                        "I found multiple leads with that name. "
                        "Please share the phone number or email of the lead you want to update."
                    )
                    asked_questions[key] = q_text
                    asked_questions["_contact_disambiguation"] = True
                    return {
                        "success": False,
                        "clarifying_question": (
                            q_text + "\nHere are the matching leads:\n- " + "\n- ".join(options)
                        ),
                        "asked_questions": asked_questions,
                    }
                else:
                    options = []
                    for r in rows:
                        parts = [r.get("name") or ""]
                        if r.get("email"):
                            parts.append(f"email: {r.get('email')}")
                        if r.get("phone_number"):
                            parts.append(f"phone: {r.get('phone_number')}")
                        options.append(" | ".join(parts))

                    key = "contact"
                    q_text = (
                        "I still see multiple leads matching that info. "
                        "Please share another identifier such as the phone number or email shown in the list."
                    )
                    asked_questions[key] = q_text
                    asked_questions["_contact_disambiguation"] = True
                    return {
                        "success": False,
                        "clarifying_question": (
                            q_text + "\nHere are the matching leads:\n- " + "\n- ".join(options)
                        ),
                        "asked_questions": asked_questions,
                    }

        if not lead_row:
            return {
                "success": False,
                "error": "Lead not found (or it doesn’t belong to you).",
            }

        resolved_id = lead_row["id"]
        resolved_name = lead_row.get("name") or lead_name or "this lead"

        # Clear identification helpers if we just disambiguated by contact info
        if asked_questions.pop("_contact_disambiguation", None):
            lead_email = None
            lead_phone = None
            asked_questions.pop("update_lead_contact", None)

        # --- 4) Ask what to update if no fields yet ---
        has_update_fields = any(
            [
                lead_name is not None,
                lead_email is not None,
                lead_phone is not None,
                platform is not None,
                status is not None,
                remarks is not None,
                follow_up is not None,
            ]
        )

        # Normalize comparison to avoid false positives on case/whitespace
        db_name = str(lead_row.get("name") or "").strip().lower()
        new_name = str(lead_name or "").strip().lower()

        if has_update_fields and new_name == db_name:
            has_update_fields = any(
                [
                    lead_email is not None,
                    lead_phone is not None,
                    platform is not None,
                    status is not None,
                    remarks is not None,
                    follow_up is not None,
                ]
            )

        # Check if user selected a field to update but hasn't provided the new value yet
        updating_field = asked_questions.get("_updating_field")
        
        # PRIORITIZE the selected field even if has_update_fields is True
        # This handles the case where lead_name was already set to the old name
        if updating_field:
            # Map the field identifier to the actual question and target field
            field_questions = {
                "update_field_name": ("lead_name", "What is the new name for this lead?"),
                "update_field_email": ("lead_email", "What is the new email address for this lead?"),
                "update_field_phone": ("lead_phone", "What is the new phone number for this lead?"),
                "update_field_platform": ("platform", "What is the new platform for this lead?"),
                "update_field_status": ("status", "What is the new status for this lead? (new, contacted, responded, qualified, converted, lost)"),
                "update_field_remarks": ("remarks", "What are the new remarks for this lead?"),
                "update_field_follow_up": ("follow_up", "What is the new follow-up date for this lead?"),
            }
            
            if updating_field in field_questions:
                target_field, question_text = field_questions[updating_field]
                key = f"update_lead_{target_field}"
                
                # Check if we've already asked for this field's value
                if key in asked_questions:
                    # User has answered, check if we have the value now
                    current_value = None
                    if target_field == "lead_name":
                        current_value = lead_name
                    elif target_field == "lead_email":
                        current_value = lead_email
                    elif target_field == "lead_phone":
                        current_value = lead_phone
                    elif target_field == "platform":
                        current_value = platform
                    elif target_field == "status":
                        current_value = status
                    elif target_field == "remarks":
                        current_value = remarks
                    elif target_field == "follow_up":
                        current_value = follow_up
                    
                    if current_value is None or (target_field == "remarks" and current_value == ""):
                        # Still waiting for answer
                        return {
                            "success": False,
                            "clarifying_question": (
                                f"I still need the {target_field.replace('_', ' ')} for this update. "
                                f"Previously I asked: '{asked_questions[key]}'"
                            ),
                            "asked_questions": asked_questions,
                        }
                    
                    # Value received, clear the _updating_field flag and continue to update logic
                    asked_questions.pop("_updating_field", None)
                    # Force has_update_fields to True so we don't show the menu again
                    has_update_fields = True 
                else:
                    # Ask for the field's new value
                    asked_questions[key] = question_text
                    options = None
                    if target_field == "platform":
                        options = ["website", "facebook", "instagram", "linkedin", "referral", "manual", "google", "twitter", "other"]
                    elif target_field == "status":
                        options = ["new", "contacted", "responded", "qualified", "converted", "lost"]
                    
                    return {
                        "success": False,
                        "clarifying_question": question_text,
                        "asked_questions": asked_questions,
                        "options": options,
                    }

        # 4) Ask what to update if no fields yet and we haven't selected a field to update
        if not has_update_fields and not updating_field:
            key = "update_lead_field"
            q = (
                f"What would you like to update for {resolved_name}? "
                "You can update the name, email, phone number, platform, status, remarks, or follow-up date."
            )
            if key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": (
                        f"I still need to know what you want to change for {resolved_name}. "
                        f"Previously I asked: '{asked_questions[key]}'"
                    ),
                    "asked_questions": asked_questions,
                    "options": [
                        "name",
                        "email",
                        "phone",
                        "platform",
                        "status",
                        "remarks",
                        "follow_up",
                    ],
                }
            asked_questions[key] = q
            return {
                "success": False,
                "clarifying_question": q,
                "asked_questions": asked_questions,
                "options": [
                    "name",
                    "email",
                    "phone",
                    "platform",
                    "status",
                    "remarks",
                    "follow_up",
                ],
            }

        # --- 5) Build update payload ---
        update_data: Dict[str, Any] = {}

        if lead_name and lead_name != lead_row.get("name"):
            update_data["name"] = lead_name
        if lead_email:
            update_data["email"] = lead_email
        if lead_phone:
            update_data["phone_number"] = lead_phone
        if platform:
            update_data["source_platform"] = platform
        if status:
            update_data["status"] = status

        metadata = lead_row.get("metadata") or {}
        if not isinstance(metadata, dict):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        if remarks is not None:
            low = str(remarks).lower().strip()
            if low in ["na", "n/a", "none"]:
                metadata.pop("remarks", None)
            else:
                metadata["remarks"] = remarks

        if follow_up is not None:
            try:
                metadata["follow_up"] = follow_up.isoformat()
            except Exception:
                metadata["follow_up"] = str(follow_up)

        if metadata != lead_row.get("metadata"):
            update_data["metadata"] = metadata

        if not update_data:
            # If we reached here and have no changes, it might be because the user provided the same value
            if updating_field:
                field_name_pretty = updating_field.replace("update_field_", "")
                return {
                    "success": True,
                    "data": {"message": f"The {field_name_pretty} is already set to that value. No update was necessary."},
                    "asked_questions": {},
                }
            
            return {
                "success": False,
                "error": "No valid changes were provided to update this lead.",
            }

        # --- 6) Apply update in Supabase ---
        logger.info(f"🔄 Updating lead {resolved_id} with data: {update_data}")
        res = (
            supabase.table("leads")
            .update(update_data)
            .eq("id", resolved_id)
            .eq("user_id", user_id)
            .execute()
        )
        updated = (res.data or [lead_row])[0]

        changed_fields = ", ".join(update_data.keys())
        msg = (
            f"✅ Updated lead '{updated.get('name') or resolved_name}'. "
            f"Updated fields: {changed_fields}."
        )

        return {
            "success": True,
            "data": {"message": msg},
            "asked_questions": {},
        }

    except Exception as e:
        logger.error(f"❌ Error updating lead in Supabase: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Database error while updating lead: {str(e)}",
        }
def _handle_search_lead(payload: LeadsManagementPayload, user_id: str, asked_questions: Dict[str, Any]) -> Dict[str, Any]:
    """Handle searching for leads"""
    name_filter = payload.lead_name
    if not name_filter and payload.key:
        name_filter = payload.key.strip()

    search_criteria = []
    if name_filter:
        search_criteria.append(f"name: {name_filter}")
    if payload.lead_email:
        search_criteria.append(f"email: {payload.lead_email}")
    if payload.lead_phone:
        search_criteria.append(f"phone: {payload.lead_phone}")

    if not search_criteria:
        question_key = "search_criteria"
        question_text = "How would you like to search for leads? Please provide a name, email, or phone number."

        if question_key in asked_questions:
            return {
                "success": False,
                "clarifying_question": f"I already asked for search criteria. Please provide a name, email, or phone number to search. Previously asked: '{asked_questions[question_key]}'",
                "asked_questions": asked_questions
            }

        asked_questions[question_key] = question_text
        return {
            "success": False,
            "clarifying_question": question_text,
            "asked_questions": asked_questions
        }

    try:
        from supabase import create_client
        import os

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found for search_lead")
            return {
                "success": False,
                "error": "Database configuration error. Please contact support."
            }

        supabase = create_client(supabase_url, supabase_key)

        query = supabase.table("leads").select(
            "id,name,email,phone_number,status,source_platform,created_at,updated_at,metadata"
        ).eq("user_id", user_id)

        if payload.lead_id:
            query = query.eq("id", payload.lead_id)
        if name_filter:
            query = query.ilike("name", f"%{name_filter}%")
        if payload.lead_email:
            query = query.eq("email", payload.lead_email)
        if payload.lead_phone:
            query = query.eq("phone_number", payload.lead_phone)

        result = query.order("created_at", desc=True).limit(5).execute()
        rows = result.data or []

        if not rows:
            criteria_text = ", ".join(search_criteria)
            return {
                "success": False,
                "error": f"No leads found matching {criteria_text}. Would you like to try another search or add a new lead?",
                "asked_questions": asked_questions
            }

        formatted_leads = []
        for lead in rows:
            lead_id = lead.get("id", "unknown")
            lead_name = lead.get("name") or "Unnamed lead"
            formatted_leads.append({
                "id": lead_id,
                "name": lead_name,
                "status": lead.get("status"),
                "platform": lead.get("source_platform"),
                "email": lead.get("email"),
                "phone_number": lead.get("phone_number"),
            })
        selected_lead = None
        if rows:
            top = rows[0]
            selected_lead = {
                "lead_id": top.get("id"),
                "lead_name": top.get("name"),
                "lead_email": top.get("email"),
                "lead_phone": top.get("phone_number"),
                "platform": top.get("source_platform"),
                "status": top.get("status"),
                "lead_status": top.get("status"),
            }

        message_lines = ["I found the lead(s). Here are the details:"]
        for idx, lead in enumerate(formatted_leads, start=1):
            message_lines.append(f"{idx}. Name: {lead['name']}")
            platform = lead.get("platform") or "platform not specified"
            status = lead.get("status") or "status not specified"
            message_lines.append(f"   Platform: {platform}")
            message_lines.append(f"   Status: {status}")
            if lead.get("email"):
                message_lines.append(f"   Email: {lead['email']}")
            if lead.get("phone_number"):
                message_lines.append(f"   Phone: {lead['phone_number']}")
            message_lines.append("")
        if message_lines and message_lines[-1] == "":
            message_lines.pop()

        data_payload = {
            "message": "\n".join(message_lines),
            "leads": formatted_leads
        }
        if selected_lead:
            data_payload["selected_lead"] = selected_lead
        return {
            "success": True,
            "data": data_payload,
            "asked_questions": {}
        }

    except Exception as e:
        logger.error(f"❌ Error searching leads in Supabase: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Search failed due to a database error: {str(e)}"
        }

def _handle_delete_lead(payload: LeadsManagementPayload, user_id: str, asked_questions: Dict[str, Any]) -> Dict[str, Any]:
    """Handle deleting a lead (requires confirmation)."""
    lead_id = payload.lead_id or asked_questions.get("identified_lead_id")
    lead_name = payload.lead_name or (getattr(payload.delete_lead, "lead_name", None) if payload.delete_lead else None)
    lead_email = payload.lead_email
    lead_phone = payload.lead_phone
    
    # Require lead identifier (id or name)
    if not lead_id and not lead_name:
        question_key = "lead_id_delete"
        question_text = "Which lead should I delete? Please provide the lead ID or name."
        if question_key in asked_questions:
            return {
                "success": False,
                "clarifying_question": f"I already asked which lead to delete. Please provide the lead ID or name to continue. Previously asked: '{asked_questions[question_key]}'",
                "asked_questions": asked_questions
            }
        asked_questions[question_key] = question_text
        return {
            "success": False,
            "clarifying_question": question_text,
            "asked_questions": asked_questions
        }

    # ACTUAL SUPABASE DELETE IMPLEMENTATION
    try:
        from supabase import create_client
        import os

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found")
            return {
                "success": False,
                "error": "Database configuration error. Please contact support."
            }

        supabase = create_client(supabase_url, supabase_key)

        # Resolve lead row
        lead_row = None

        if lead_id:
            res = supabase.table("leads").select("id,name,email,phone_number").eq("id", lead_id).eq("user_id", user_id).limit(1).execute()
            if res.data:
                lead_row = res.data[0]
                asked_questions["identified_lead_id"] = lead_row["id"]
        else:
            # Name-based search (may be ambiguous)
            res = supabase.table("leads").select("id,name,email,phone_number").eq("user_id", user_id).ilike("name", f"%{lead_name}%").limit(10).execute()
            rows = res.data or []
            if len(rows) == 1:
                lead_row = rows[0]
                asked_questions["identified_lead_id"] = lead_row["id"]
                verified_contact = lead_email or lead_phone
                if verified_contact:
                    asked_questions["contact_verified"] = verified_contact
                    asked_questions.pop("contact_verification_pending", None)
                asked_questions.pop("contact", None)
            elif len(rows) > 1:
                # Try to narrow down using provided contact info (email/phone)
                filtered = []
                if lead_email or lead_phone:
                    for r in rows:
                        if lead_email and r.get("email") and r.get("email").lower() == lead_email.lower():
                            filtered.append(r)
                        elif lead_phone and r.get("phone_number") and r.get("phone_number") == lead_phone:
                            filtered.append(r)
                if len(filtered) == 1:
                    lead_row = filtered[0]
                    asked_questions["identified_lead_id"] = lead_row["id"]
                    verified_contact = lead_email or lead_phone
                    if verified_contact:
                        asked_questions["contact_verified"] = verified_contact
                        asked_questions.pop("contact_verification_pending", None)
                    asked_questions.pop("contact", None)
                else:
                    choices = []
                    for r in rows:
                        details = []
                        if r.get("email"):
                            details.append(f"email: {r.get('email')}")
                        if r.get("phone_number"):
                            details.append(f"phone: {r.get('phone_number')}")
                        info = "; ".join(details) if details else "no additional info"
                        choices.append(f"{r.get('name')} ({info})")

                    question_text = (
                        "I found multiple leads with that name. "
                        "Please share the phone number or email of the lead you want to delete."
                    )
                    asked_questions["contact"] = question_text
                    asked_questions["contact_verification_pending"] = True
                    return {
                        "success": False,
                        "clarifying_question": (
                            question_text + "\nHere are the matching leads:\n- " + "\n- ".join(choices)
                        ),
                        "asked_questions": asked_questions
                    }

        if not lead_row:
            return {
                "success": False,
                "error": "Lead not found (or it doesn’t belong to you)."
            }

        verified_contact = asked_questions.get("contact_verified")
        if asked_questions.get("contact_verification_pending") and not verified_contact:
            return {
                "success": False,
                "clarifying_question": "I’m still waiting for the phone number or email that matches one of the leads I found. Please share that so I can identify the correct lead.",
                "asked_questions": asked_questions
            }

        if payload.confirm_delete is not True:
            question_key = "confirm_delete"
            target_label = lead_row.get("name") or lead_id or "this lead"
            verification_note = f" (verified via {verified_contact})" if verified_contact else ""
            question_text = f"Are you sure you want to delete {target_label}{verification_note}? (yes/no)"
            if question_key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": f"I already asked for confirmation. Please reply **yes** to delete or **no** to cancel. Previously asked: '{asked_questions[question_key]}'",
                    "options": ["yes", "no"],
                    "asked_questions": asked_questions
                }
            asked_questions[question_key] = question_text
            return {
                "success": False,
                "clarifying_question": question_text,
                "options": ["yes", "no"],
                "asked_questions": asked_questions
            }

        lead_id = lead_row.get("id")

        # Delete related data first
        supabase.table("lead_status_history").delete().eq("lead_id", lead_id).execute()
        supabase.table("lead_conversations").delete().eq("lead_id", lead_id).execute()

        # Delete the lead itself (scoped by user_id for safety)
        supabase.table("leads").delete().eq("id", lead_id).eq("user_id", user_id).execute()

        return {
            "success": True,
            "data": {
                "message": f"✅ Deleted lead successfully: {lead_row.get('name') or ''}"
            },
            "asked_questions": {}  # Clear asked_questions after success
        }

    except Exception as e:
        logger.error(f"❌ Error deleting lead in Supabase: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }

def _handle_insight(payload: LeadsManagementPayload, user_id: str, asked_questions: Dict[str, Any]) -> Dict[str, Any]:
    """Handle lead insights query - Prioritized handler for aggregated metrics."""
    from collections import Counter
    from datetime import datetime, timedelta, timezone

    # Extract insight-specific fields (from nested payload or flattened fields)
    lead_id = payload.lead_id
    lead_name = payload.lead_name
    platform = payload.platform
    date_range = payload.date_range
    insight_type = payload.insight_type or payload.status_type  # Prefer insight_type, fallback to status_type

    if payload.insight:
        lead_id = payload.insight.lead_id or lead_id
        lead_name = payload.insight.lead_name or lead_name
        platform = payload.insight.platform or platform
        date_range = payload.insight.date_range or date_range
        insight_type = payload.insight.insight_type or insight_type

    # Build filter context description
    context_fragments = []
    if lead_id:
        context_fragments.append(f"ID {lead_id}")
    if lead_name:
        context_fragments.append(f"name '{lead_name}'")
    if platform:
        context_fragments.append(f"platform '{platform}'")
    context_description = " and ".join(context_fragments) if context_fragments else "all leads"

    try:
        from supabase import create_client
        import os

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found for insights")
            return {
                "success": False,
                "error": "Database configuration error. Please contact support."
            }

        supabase = create_client(supabase_url, supabase_key)

        query = supabase.table("leads").select(
            "id,name,status,source_platform,created_at"
        ).eq("user_id", user_id)

        if platform:
            query = query.eq("source_platform", platform)
        if lead_id:
            query = query.eq("id", lead_id)
        elif lead_name:
            query = query.ilike("name", f"%{lead_name}%")

        response = query.order("created_at", desc=True).limit(2000).execute()
        leads = response.data or []

        if not leads:
            return {
                "success": False,
                "error": f"No leads found for {context_description}.",
                "asked_questions": asked_questions
            }

        def _parse_created_at(created_at_value: Optional[str]) -> Optional[datetime]:
            if not created_at_value:
                return None
            iso_ts = created_at_value
            if iso_ts.endswith("Z"):
                iso_ts = iso_ts.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(iso_ts)
            except ValueError:
                try:
                    parsed = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%S.%f%z")
                except Exception:
                    return None
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed

        now = datetime.utcnow()
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_week = start_today - timedelta(days=start_today.weekday())
        start_month = start_today.replace(day=1)
        start_prev_week = start_week - timedelta(weeks=1)

        status_counter: Counter[str] = Counter()
        platform_counter: Counter[str] = Counter()
        leads_this_day = leads_this_week = leads_this_month = 0
        current_week_count = previous_week_count = 0

        for lead in leads:
            created_at = _parse_created_at(lead.get("created_at"))
            if not created_at:
                continue

            if created_at >= start_today:
                leads_this_day += 1
            if created_at >= start_week:
                leads_this_week += 1
            if created_at >= start_month:
                leads_this_month += 1

            if created_at >= start_week:
                current_week_count += 1
            elif created_at >= start_prev_week:
                previous_week_count += 1

            status = lead.get("status")
            if status:
                status_counter[status] += 1

            source_platform = lead.get("source_platform")
            if source_platform:
                platform_counter[source_platform] += 1

        total_leads = len(leads)

        status_breakdown = [
            {"status": status, "count": count}
            for status, count in status_counter.most_common()
            if count > 0
        ]

        platform_breakdown = [
            {"platform": platform_name, "count": count}
            for platform_name, count in platform_counter.most_common()
        ]

        trend_difference = current_week_count - previous_week_count
        if previous_week_count == 0 and current_week_count == 0:
            trend_state = "steady"
            trend_summary = "No leads were created this week or last week."
        elif current_week_count >= previous_week_count:
            trend_state = "improving"
            trend_summary = (
                f"You're building positive momentum: {current_week_count} leads this week "
                f"versus {previous_week_count} last week (+{trend_difference})."
            )
        else:
            trend_state = "declining"
            decline_diff = abs(trend_difference)
            trend_summary = (
                f"Activity has dipped: {current_week_count} leads this week compared to "
                f"{previous_week_count} last week (-{decline_diff})."
            )

        insight_payload = {
            "options": [
                "Total leads",
                "Leads by status",
                "Leads by platform",
                "Time-based trends"
            ],
            "total_leads": {
                "total": total_leads,
                "today": leads_this_day,
                "this_week": leads_this_week,
                "this_month": leads_this_month
            },
            "status_breakdown": status_breakdown,
            "platform_breakdown": platform_breakdown,
            "time_trends": {
                "current_week": current_week_count,
                "previous_week": previous_week_count,
                "difference": trend_difference,
                "trend": trend_state,
                "summary": trend_summary
            }
        }

        if platform_breakdown:
            top_platform = platform_breakdown[0]
            insight_payload["platform_highlight"] = {
                "platform": top_platform["platform"],
                "count": top_platform["count"],
                "message": f"Top platform: {top_platform['platform']} ({top_platform['count']} leads)"
            }

        options_text = ", ".join(insight_payload["options"])
        insight_type_normalized = None
        if insight_type:
            insight_type_normalized = insight_type.strip().lower()
            if "time" in insight_type_normalized and "trend" in insight_type_normalized:
                insight_type_normalized = "time_trends"
        panel_cards = {
            "summary": {
                "label": "Total leads",
                "data": insight_payload["total_leads"],
                "message": (
                    f"📊 Total leads {total_leads} · Today {leads_this_day} · Week {leads_this_week} · Month {leads_this_month}"
                )
            },
            "status": {
                "label": "Leads by status",
                "data": status_breakdown,
                "message": (
                    "📋 Lead statuses:\n" +
                    "\n".join(
                        f"- {entry['status']}: {entry['count']}"
                        for entry in status_breakdown
                    ) if status_breakdown else "No status data available."
                )
            },
            "platform": {
                "label": "Leads by platform",
                "data": platform_breakdown,
                "message": (
                    "📍 Leads sorted by platform:\n" +
                    "\n".join(
                        f"- {entry['platform']}: {entry['count']}"
                        for entry in platform_breakdown
                        if entry["count"]
                    ) if platform_breakdown else "Platform data is not available yet."
                )
            },
            "time_trends": {
                "label": "Time-based trends",
                "data": insight_payload["time_trends"],
                "message": (
                    f"{trend_summary}"
                )
            }
        }

        selected_panel = panel_cards.get(insight_type_normalized) if insight_type_normalized else None
        if selected_panel:
            message = selected_panel["message"]
        else:
            message = (
                f"Here are the lead insights for {context_description}. Choose a panel you’d like to explore: "
                f"{options_text}."
            )

        insight_payload["panels"] = panel_cards
        insight_payload["selected_panel"] = insight_type_normalized if selected_panel else None
        insight_payload["options_text"] = options_text

        insight_context = {}
        if lead_id:
            insight_context["lead_id"] = lead_id
        if lead_name:
            insight_context["lead_name"] = lead_name
        if platform:
            insight_context["platform"] = platform

        return {
            "success": True,
            "data": {
                "message": message,
                "insight_type": insight_type or "summary",
                "date_range": date_range,
                "insights": insight_payload,
                "insight_context": insight_context
            },
            "asked_questions": {}
        }

    except Exception as e:
        logger.error(f"❌ Error generating insights: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Database error while generating insights: {str(e)}"
        }

