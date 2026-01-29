from pydantic import BaseModel, Field
from typing import List, Optional

class ContractParties(BaseModel):
    party_a: str = Field(..., description="The full legal name of the first party (e.g., Service Provider).")
    party_b: str = Field(..., description="The full legal name of the second party (e.g., Client).")

class ContractDates(BaseModel):
    effective_date: Optional[str] = Field(None, description="The start date of the contract in YYYY-MM-DD format. If 'Upon Signature', explicitly state that.")
    expiration_date: Optional[str] = Field(None, description="The end date or duration (e.g., '1 Year from Effective Date').")

class PaymentDetails(BaseModel):
    total_fee: Optional[str] = Field(None, description="The total value or fee mentioned. Include currency symbols (e.g., '$5,000').")
    payment_terms: Optional[str] = Field(None, description="When the payment is due (e.g., 'Net 30', 'Upon receipt').")

class DocumentReviewOutput(BaseModel):
    summary_simple: str = Field(..., description="A 2-3 sentence summary of the document written in simple, non-legal English for a layman.")
    parties: ContractParties
    dates: ContractDates
    financials: PaymentDetails
    risk_flags: List[str] = Field(default=[], description="List any missing critical clauses (e.g., 'No Termination Clause found').")