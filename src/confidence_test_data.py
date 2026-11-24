
confidence_test_emails = [

    # HR — 5 emails
    {
        "subject": "Monthly salary not credited",
        "body": "My salary for October has not been credited. Please check with the payroll team.",
        "actual": "HR",
        "expected_review": False
    },
    {
        "subject": "Employee laptop discount eligibility",
        "body": "I am a Nexora employee. Am I eligible for the employee discount when purchasing a company laptop?",
        "actual": "HR",
        "expected_review": False
    },
    {
        "subject": "Request for leave approval",
        "body": "I need leave for three days next week due to a family function. Please approve my leave request.",
        "actual": "HR",
        "expected_review": False
    },
    {
        "subject": "Incorrect deduction in my salary",
        "body": "There is an unexpected deduction in my salary. I do not know what it is for. Please look into this.",
        "actual": "HR",
        "expected_review": False
    },
    {
        "subject": "Need help with access",
        "body": "I recently joined the company and cannot access the required system. Please help me.",
        "actual": "HR",
        "expected_review": True
        
    },

    # ADMIN — 5 emails
    {
        "subject": "Office Wi-Fi connectivity issue",
        "body": "The office Wi-Fi is not working on the third floor. Please arrange technical support.",
        "actual": "ADMIN",
        "expected_review": False
    },
    {
        "subject": "Stationery supplies required",
        "body": "We need additional printer paper, pens and office files. Please arrange procurement.",
        "actual": "ADMIN",
        "expected_review": False
    },
    {
        "subject": "Courier tracking for office documents",
        "body": "The courier containing signed internal documents is delayed. Please contact the courier provider.",
        "actual": "ADMIN",
        "expected_review": False
    },
    {
        "subject": "Laptop required for new employee",
        "body": "A new employee is joining next Monday. Please arrange a laptop and necessary equipment.",
        "actual": "ADMIN",
        "expected_review": False
    },
    {
        "subject": "Please look into the attached document",
        "body": "Please check the attached file and take the necessary action.",
        "actual": None,
        "expected_review": True
        
    },

    # LEGAL — 5 emails
    {
        "subject": "Formal legal notice regarding unpaid dues",
        "body": "Our legal representatives hereby issue a formal notice demanding payment of outstanding dues within seven days.",
        "actual": "LEGAL",
        "expected_review": False
    },
    {
        "subject": "Contract termination dispute",
        "body": "Our legal counsel disputes the termination of the agreement and demands compliance with the contractual obligations.",
        "actual": "LEGAL",
        "expected_review": False
    },
    {
        "subject": "Cease and desist notice",
        "body": "Our lawyers demand that Nexora immediately stop the unauthorised use of our registered trademark.",
        "actual": "LEGAL",
        "expected_review": False
    },
    {
        "subject": "Please review the attached agreement",
        "body": "The attached agreement contains several disputed clauses. Our legal advisor has requested a review before signing.",
        "actual": "LEGAL",
        "expected_review": False
    },
    {
        "subject": "We may take further action",
        "body": "We are unhappy with the situation and may take further action if this is not resolved.",
        "actual": None,
        "expected_review": True
    },

    # SALES — 5 emails
    {
        "subject": "Warranty replacement for damaged television",
        "body": "My Nexora television arrived with a cracked screen. Please arrange a warranty replacement.",
        "actual": "SALES",
        "expected_review": False
    },
    {
        "subject": "Refund pending for cancelled order",
        "body": "I cancelled my Nexora order ten days ago, but the refund has not been credited. Please check.",
        "actual": "SALES",
        "expected_review": False
    },
    {
        "subject": "Bulk quotation for laptops",
        "body": "Our organisation wants to purchase 100 Nexora laptops. Please share the bulk quotation and delivery details.",
        "actual": "SALES",
        "expected_review": False
    },
    {
        "subject": "Device stopped working",
        "body": "The device I purchased recently has stopped working. Please help me resolve this issue.",
        "actual": "SALES",
        "expected_review": False
    },
    {
        "subject": "Urgent issue with my order",
        "body": "I have an issue with my recent order. Please contact me as soon as possible.",
        "actual": "SALES",
        "expected_review": False
    }
]
