"""
Example of using LangChain models with browser-use.

This example demonstrates how to:
1. Wrap a LangChain model with ChatLangchain
2. Use it with a browser-use Agent
3. Run a simple web automation task

@file purpose: Example usage of LangChain integration with browser-use
"""

import asyncio
import os

from langchain_openai import ChatOpenAI  # pyright: ignore

from browser_use import Agent
from fix.chat_qwen import ChatLangchain

API_KEY = os.getenv("API_KEY", "")

BASE_URL = os.getenv("BASE_URL", "")
# BASE_URL = os.getenv("BASE_URL_SECUNDARIO", "")


MODELO = 'qwen3-coder:30b'
# MODELO = 'Qwen3-Coder-30B-A3B'
# MODELO = 'Qwen3-VL-32B'

async def main():
	"""Basic example using ChatLangchain with OpenAI through LangChain."""

	langchain_model = ChatOpenAI(
		base_url=BASE_URL,
        model=MODELO,
        api_key=API_KEY,
		temperature=0.0,
		max_tokens=4096,
	)

	llm = ChatLangchain(chat=langchain_model)

	# task = "Acesse 'http://localhost:9990/en_US/', acesse opção de 'T-shirts' e acesse as opções disponíveis para  'women'"

	# task = "Go to google.com and search for 'browser automation with Python'"


	task = """
		Context: You are a Senior QA Engineer executing an automated regression test on the Sylius application.
        Start URL: http://localhost:9990/en_US/

        ⚠️ PRIMARY DIRECTIVE (READ CAREFULLY):
        Your role is NOT to make the test pass. Your role is to REPORT THE TRUTH.
        If the value on the screen is different from expected, this is an application BUG, not your error.
        NEVER try to read again to "find" the expected value. Accept the screen value on the first read as the Absolute Truth.

        Objective: Validate the complete Product Purchase flow (E2E).

        Execution Steps:
        1. Navigate to the main page at http://localhost:9990/en_US/.
        2. Identify the item 'T-shirts' and select the option for women 'women'.
        3. Select the option 'Everyday white basic T-Shirt'.
        4. Select the field 'T-shirt size' and select option 'M'.
        5. Fill the 'Quantity' with the value 5.
        6. Click on the button 'Add to cart' ONLY AFTER 'T-shirt size' are filled with the value 'M' and 'Quantity' are filled with the value '5'.
        7. Click on the button 'Checkout'.
        8. Fill the 'Email' field with the value 'browser.use@gmail.com'.
        9. Fill the 'First name' field with the value 'browser'.
        10. Fill the 'Last name' field with the value 'use'.
        11. Fill the 'Street address' field with the value 'browser.use'.
        12. Click on the 'Country' field and select 'Australia'.
        13. Fill the 'City' field with the value 'browser.use'.
        14. Fill the 'Postcode' field with the value '123456789'.
        15. Click on the button 'Next' ONLY AFTER 'City' and 'Postcode' are filled.
        16. In the 'Shipment #1' field select the option 'FedEx'.
        17. Click on the button 'Next'.
        18. In the 'Payment #1' field select the option 'Cash on delivery'.
        19. Click on the button 'Next'.

        Acceptance Criteria (Strict Validation):
        1. [BILLING_ADDRESS] VERIFY if the address information ('browser use', 'browser.use', 'browser.use, 123456789', 'AUSTRALIA') appeared.
        2. [SHIPPING_ADDRESS] VERIFY if the address information ('browser use', 'browser.use', 'browser.use, 123456789', 'AUSTRALIA') appeared.
            
        3. [ITEM] VERIFY if 'Item' is 'Everyday white basic T-Shirt'.
        4. [ITEM] VERIFY if 'Item Everyday white basic T-Shirt' is 'M'.
            
        5. [UNIT_PRICE] VERIFY if 'Unit price' is exactly '$94.60'.
        6. [QUANTITY] VERIFY if 'Qty' is exactly '5'.
        7. [SUBTOTAL] VERIFY if 'Subtotal' is exactly '$473.00'.
        8. [SHIPPING_TOTAL] VERIFY if 'Shipping total:' is exactly '$3.44'.
        9. [TAXES_TOTAL] VERIFY if 'Taxes total:' is exactly '$0.00'.
        10. [DISCOUNT] VERIFY if 'Discount:' is exactly '$0.00'.
        11. [TOTAL_FINAL] VERIFY if 'Total:' is exactly '$476.44'.
        12. [CASH_DELIVERY] VERIFY if 'Cash on delivery' is exactly '$476.44'.

        ---
        ### 🧠 THOUGHT PROCESS, NORMALIZATION AND LINKING (CoT)
        Before generating the table line, follow this RIGID mental flow:

        1. APPLY CLEANING RULES (Mentally):
        - **RULE 1 (Text/Messages):** Ignore extra spaces (`trim`) and line breaks (`\n`). If the clean message CONTAINS the expected keywords, consider it SUCCESS.
            *Ex: "  Success\nItem... " == "Success Item..." -> ✅ APPROVED.*
        - **RULE 2 (Numbers/Prices):** EXACT comparison. '$473.00' is different from '$463.00'. This is a BUG.
        - **RULE 3 (Visual Sanity):** If 'Expected' and 'Actual' look IDENTICAL to the human eye, but the code says different (ex: invisible character), APPROVE THE ITEM.

        2. GENERATE DIAGNOSTIC TEXT (First writing step):
        - If passed the rules above: The text MUST start with **"OK:"**.
        - If failed the rules above: The text MUST start with **"ERROR:"**.

        3. DEFINE STATUS (Last step - Linking):
        - Look at the text you just generated.
        - If the text starts with "OK:" -> The STATUS icon is MANDATORILY **'✅'**.
        - If the text starts with "ERROR:" -> The STATUS icon is MANDATORILY **'❌'**.

        **Example of Correct Logic:**
        - *Thought:* "The message has extra spaces, but contains the text."
        - *Diagnostic:* "OK: Text found (with spaces)."
        - *Status:* ✅ (Because the diagnostic is OK).

        ⛔ FORBIDDEN:
        - Do NOT perform a second extraction if values after CLEANING RULE application do not match.
        - Do NOT assume you read wrong after CLEANING RULE application.
        - Do NOT enter a loop. If there is a discrepancy after CLEANING RULE application, JUST report it.
        - Never put '❌' if the text says "OK". Never put '✅' if the text says "ERROR".
        
        ---

        ### 🛑 TERMINATION PROTOCOL (ANTI-LOOP) - READ CAREFULLY
        You are prohibited from using the tool 'extract' more than once.
        As soon as you have the data in the log (even if they are wrong or different from expected):

        1.  **STOP** using tools. Do not try to navigate, click or extract again.
        2.  **THERE IS NO** validation tool. Validation is a mental process of yours.
        3.  **THERE IS NO** report tool. The report is just TEXT.
        4.  Your next action must be **"Final Answer"** (Final Answer) containing ONLY the formatted report below.

        If you call the tool 'extract' for the second time, you FAILED the mission.

        ### 🖥️ TERMINAL OUTPUT CONFIGURATION (MANDATORY)
        Upon having the data, fill this template and return it as final text:
        Generate a report visually formatted for monospace console logs.
        Do NOT use complex Markdown tables. Use the aligned list format below.
        The display order must be: 1st Details -> 2nd Final Status.
        Do NOT print 'TABLE FILLING RULES' at the end of execution.

        Follow EXACTLY this visual template:

        📋 EXECUTION DETAILS (ASSERTIONS):
        --------------------------------------------------------------------------------
        [STATUS] | ID  | OBJECT/ACTION             | DIAGNOSTIC (EXPECTED vs ACTUAL)
        --------------------------------------------------------------------------------
        [{STATUS}]  | {ID}  | {SHORT_CRITERIA_NAME}     | {MSG_RESULT}
        ... (repeat this line for each validation performed) ...
        --------------------------------------------------------------------------------

        ================================================================================
        🚀 AUTOMATED EXECUTION REPORT: SYLIUS E-COMMERCE
        ================================================================================
        📅 DATE: [Current Date/Time]
        🎯 SCENARIO: Purchase Flow (End-to-End)
        🏁 FINAL STATUS: [ APPROVED ✅ / FAILED ❌ ]
        ================================================================================

        TABLE FILLING RULES:
        1. {STATUS}: Use '✅' for success or '❌' for failure.
        2. {ID}: Sequential number (01, 02, 03...).
        3. {SHORT_CRITERIA_NAME}: Summarize what was tested in max 25 characters (ex: "Login Button", "Shipping Calc", "Redirect URL").
        4. {STATUS} (MANDATORY LINKING):
           - Do NOT make a new comparison here. Look only at the {MSG_RESULT} you defined.
           - If {MSG_RESULT} starts with "OK" -> Use STRICTLY '✅'.
           - If {MSG_RESULT} starts with "ERROR" -> Use STRICTLY '❌'. 
        5. FINAL CONSISTENCY:
           - It is IMPOSSIBLE to have '❌' with "OK".
           - It is IMPOSSIBLE to have '✅' with "ERROR".
           - In doubt, trust the text "OK/ERROR" and adjust the icon to match.
        6. {MSG_RESULT} (In case of failure): It is MANDATORY to show the real value. Ex: "ERROR: Exp '$479.68' | Actual '$0.00' (or 'Not found')".
	"""

	agent = Agent(
		task=task,
		llm=llm,
		use_vision=False,
	)

	print(f'🚀 Starting task: {task}')
	print(f'🤖 Using model: {llm.name} (provider: {llm.provider})')

	history = await agent.run()

	print(f'✅ Task completed! Steps taken: {len(history.history)}')

	if history.final_result():
		print(f'📋 Final result: {history.final_result()}')

		return history


if __name__ == '__main__':
	print('🌐 Browser-use LangChain Integration Example')
	print('=' * 45)

	asyncio.run(main())