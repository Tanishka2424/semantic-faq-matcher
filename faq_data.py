"""Shared FAQ data used by app.py and eval.py."""

# Each FAQ has its canonical question plus alternate phrasings ("alts").
# Retrieval scores a query against ALL phrasings and keeps the best per FAQ.
# NOTE: write alts WITHOUT looking at your test set, otherwise your accuracy number is inflated.
FAQS = [
    {"q": "How can I track my order?",
     "a": "Go to My Orders and click Track. You will also get SMS/email updates.",
     "alts": ["Where is my shipment?", "Order status and tracking", "I want to know where my delivery is"]},
    {"q": "What is your return policy?",
     "a": "You can return items within 7 days of delivery if they are unused.",
     "alts": ["Can I return a product?", "How do I send an item back?", "I am not happy with what I bought, can I give it back?"]},
    {"q": "How do I reset my password?",
     "a": "Click Forgot Password on the login page and follow the email link.",
     "alts": ["I cannot log in to my account", "Change my account password", "I lost my login credentials"]},
    {"q": "What payment methods do you accept?",
     "a": "We accept UPI, credit/debit cards, net banking and wallets.",
     "alts": ["How can I pay?", "Which cards or UPI apps are supported?", "Accepted modes of payment"]},
    {"q": "How long does delivery take?",
     "a": "Delivery usually takes 3-5 working days.",
     "alts": ["When will my order arrive?", "What is the expected delivery time?", "How many days for shipping?"]},
    {"q": "Can I cancel my order?",
     "a": "Yes, you can cancel from My Orders before the item is shipped.",
     "alts": ["I changed my mind, stop my purchase", "How do I cancel what I ordered?", "Cancel an order that has not shipped"]},
    {"q": "How do I contact customer support?",
     "a": "Email support@shopeasy.example or call 1800-000-000 (9am-6pm).",
     "alts": ["Customer care phone number", "I need to speak to a human", "How can I reach your help desk?"]},
    {"q": "Do you offer cash on delivery?",
     "a": "Yes, cash on delivery is available for orders under Rs. 5000.",
     "alts": ["Can I pay in cash at the door?", "Is COD available?", "Pay on receipt of goods"]},
    {"q": "How do I get a refund?",
     "a": "Refunds are credited to your original payment method in 5-7 days.",
     "alts": ["When will I be reimbursed?", "I want my money returned", "How long does a refund take?"]},
    {"q": "Do you ship internationally?",
     "a": "Currently we ship only within India.",
     "alts": ["Can you deliver to other countries?", "Overseas shipping available?", "Do you deliver abroad?"]},
    {"q": "How can I change my delivery address?",
     "a": "Edit the address in My Orders before the order is shipped.",
     "alts": ["I typed the wrong address", "Update shipping address", "Deliver to a different location"]},
    {"q": "Are there any discount coupons available?",
     "a": "Check the Offers page for active coupon codes.",
     "alts": ["Do you have promo codes?","Any sales or deals right now?", "How do I get a discount?"]},
]

EXAMPLES = [
    "Where is my package right now?",
    "I forgot my login password",
    "Can I pay when the product arrives?",
    "Can I send back a product I don't like?",
    "What is the capital of France?",
]


# Also index each FAQ's answer text as an extra phrasing (helps queries that match the answer,
# e.g. "support hours"). Set to False to measure the effect without it.
ADD_ANSWER_AS_PHRASING = True
if ADD_ANSWER_AS_PHRASING:
    for _f in FAQS:
        _f["alts"] = _f["alts"] + [_f["a"]]
