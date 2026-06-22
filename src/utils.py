import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def trigger_security_alert(transaction_payload, risk_score):
    """
    Simulates localized enterprise security alerts when a transaction
    violates safe-boundary variance parameters.
    """
    message = f"🚨 SECURITY WARNING: High-risk Transaction Identified! Risk Profile: {risk_score:.2%}"
    logging.warning(message)
    
    # Mocking standard business email automation protocols
    mock_email_payload = f"""
    Subject: [CRITICAL ALERT] FRAUD MITIGATION RISK ASSIGNMENT
    To: RiskOperations@enterprise.com
    
    Automated triggers caught an outlying operational sequence:
    Transaction Context: {transaction_payload}
    Calculated Risk Vector: {risk_score:.4f}
    Recommended Action: Suspend associated credentials immediately.
    """
    return message, mock_email_payload