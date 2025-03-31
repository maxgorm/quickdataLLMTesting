import json
import os
import base64
import google.generativeai as genai
import anthropic
import sys  # To exit if no PDF or XLSX found
import pandas as pd
from dotenv import load_dotenv
from portkey_ai import Portkey

# Load environment variables from .env file
load_dotenv()

# --- API Keys (WARNING: Hardcoding keys is insecure. Use environment variables or secrets management in production) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Get from environment variable
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")  # Get from environment variable
PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")  # Get from environment variable
PORTKEY_GEMINI_VIRTUAL_KEY = os.environ.get("PORTKEY_GEMINI_VIRTUAL_KEY")  # Get from environment variable
PORTKEY_CLAUDE_VIRTUAL_KEY = os.environ.get("PORTKEY_CLAUDE_VIRTUAL_KEY")  # Get from environment variable

if not GEMINI_API_KEY:
    print("GEMINI_API_KEY environment variable not set.  Exiting.")
    sys.exit(1)

if not CLAUDE_API_KEY:
    print("CLAUDE_API_KEY environment variable not set.  Exiting.")
    sys.exit(1)

if not PORTKEY_API_KEY:
    print("PORTKEY_API_KEY environment variable not set.  Exiting.")
    sys.exit(1)

if not PORTKEY_GEMINI_VIRTUAL_KEY:
    print("PORTKEY_GEMINI_VIRTUAL_KEY environment variable not set.  Exiting.")
    sys.exit(1)

if not PORTKEY_CLAUDE_VIRTUAL_KEY:
    print("PORTKEY_CLAUDE_VIRTUAL_KEY environment variable not set.  Exiting.")
    sys.exit(1)

# --- Input/Output Directories ---
INPUT_DIR = "pdf_input"
OUTPUT_DIR = "json_output"

# --- Configure LLM Clients ---
# Initialize Portkey
try:
    portkey = Portkey(
        api_key=PORTKEY_API_KEY
    )
    print("Portkey client initialized successfully.")
except Exception as e:
    print(f"Error initializing Portkey client: {e}")
    portkey = None
    sys.exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Error configuring Gemini client: {e}")
    # Consider exiting or handling the error appropriately
    gemini_model = None

try:
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
except Exception as e:
    print(f"Error configuring Claude client: {e}")
    # Consider exiting or handling the error appropriately
    claude_client = None


# --- Core Prompt Sections ---

PROMPT_PART_1_CLAUDE = """
You are an expert real estate analyst specializing in rent roll document analysis. Your task is to examine a section of a rent roll document, accurately identify the existing column headers, and extract specific information for each unit in a token-efficient manner.

First of all you need to understand the column headers of the rent roll. Carefully analyze what is present and what is not. Also understand the data structure and read/understand all the instrcutions carefully before providing the output.

1. Rent Roll Extract:
- The rent roll extract is provided between the <RR> </RR> tags. This may be a partial section of a larger document. So it could cut off certain unit information.
- 'NULL' is used to signify blank cells in the RR. Do not include NULL in your outputs because they are just blank cells.

2. Column Header Identification:
- Carefully examine the first few rows of the data grid to identify the actual column headers.
- Be aware that column headers may span multiple rows. Combine multi-row headers into a single string, separating parts with a space.
- Only include headers that are explicitly present in the data grid. Do not infer or add headers that are not there.
"""

PROMPT_PART_2_GEMINI = """
3. Data Extraction and Processing:
- Identify Unit Blocks:
    - Recognize that information for a single unit may span multiple rows.
    - The first row of a unit block typically contains the unit identifier (e.g., unit number or letter) and primary lease information.
    - Subsequent rows without a unit identifier in the 'Unit' column are part of the same unit's data block.

- Extract information for each unit entry in the provided section, based on the identified column headers.
- Convert all dates to MM/DD/YYYY format without any time information.
- In multi row rent rolls, capture all the rows to extract information related to that unit.
- Do not try to alter or remove units or unit details or fix any mistakes. Report everything as it is.

4. Information to Extract:
- row_index: It's the row number which is also given in RR, report it as it is.
- unit_num: The unique identifier or the unit number for each rental unit as given in the rent roll.
    - unit_num can never be 'nan'.
    - Its very important you always capture the correct full unit_num even with spaces.
    - unit_num in your output units should have the same pattern as the units in RR_FIRST_30_OUTPUT.
        - Make sure to keep unit_num coherent and consistent between your current output and RR_FIRST_30_OUTPUT.
        - e.g. If RR_FIRST_30_OUTPUT has unit_num such as '608   501', you output should also have a similar type unit_num, not something like '601'.
    - For any unit numbers containing spaces. Replace those spaces with hyphen(-).
        - e.g. 'X   Y' -> 'X-Y' (correct), 'X' (incorrect), 'Y' (incorrect)
        - e.g. '608   501' -> '608-501' (correct), '608' (incorrect), '501' (incorrect)
        - e.g. '22   701' -> '22-701' (correct), '22' (incorrect), '701' (incorrect)
    - unit_num can NEVER be 'nan'.
    - Even in multi-row rent rolls, unit_num is usually in the first row of the unit rows.

- unit_type: The specific layout or type of the unit (e.g., 1BR/1BA, Studio).
    - unit_type can NEVER be 'nan'. If not directly provided, derive it accordingly.
    - It's like a code that is used to identify the type of the unit, floorplan, etc.
    - unit_type and unit_num are not the same thing. unit_type is not as unique as unit_num.

- sqft: The total square footage of the unit.

- br: The number of bedrooms in the unit.
    - Look carefully for this number in the unit. Sometimes they can be in different names.
    - Sometimes, It could be inside a breakdown. Look for words such as bed and bath.
    - Both br and bath could be in the same column like a floorplan or unit type.
    - Always give it as one decimal float.
    - If not provided within the rent roll, put 'nan' as the value.

- bath: The number of bathrooms in the unit.
    - Look carefully for this number in the unit. Sometimes they can be in different names.
    - Sometimes, It could be inside a breakdown. Look for words such as bed and bath.
    - Both br and bath could be in the same column like a floorplan or unit type.
    - Always give it as one decimal float.
    - If not provided within the rent roll, put 'nan' as the value.

- tenant: The name of the current tenant. Use 'nan' for vacant units.

- move_in: Move in date is the date the tenant moved into the unit (MM/DD/YYYY format).

- lease_start: The start date of the current lease agreement (MM/DD/YYYY format).

- lease_end: The end date of the current lease agreement (MM/DD/YYYY format).

- rent_charge: The actual base lease rent paid by the tenant.
    - This always exist in the rent roll and can NEVER be null.
    - Do not try to calculate or derive. Report as it is.
    - Do not confuse rent_charge with rent_market.
    - In case the rent roll has a charge_code_bd and it has the rent charge in it, pick this from the 'rent_charge'.
    - NEVER use a rent subsidy as 'rent_charge'. If a unit has a rent subsidy, that means 'rent_charge' is the resident's portion of the rent and not the net rent or rent subsidy.

- rent_gov_subsidy: The portion of rent covered by any government housing assistance program or government subsidy, including Section 8, Housing Choice Vouchers, HAP payments, and any federal/state housing subsidies.
    - Look in columns such as HAP rent, section 8 etc. It could also be within the charge code breakdown.
    - In case there are no rent_gov_subsidy amount, make this 'nan'.

- is_mtm: A flag indicating whether the unit is operating under a month-to-month (MTM) lease status rather than a fixed-term lease agreement.
    - Output 1 (true) to indicate MTM pricing is in effect:
        - The unit has MTM/MO/M-T-M flag or indicator in any status column
        - OR any MTM premium charges exist. i.e. 'mtm_charge' > 0
        - OR lease dates show expired/holdover status continuing as MTM.
    - Output 0 (false) if there are no MTM pricing in effect.
    - IMPORTANT: Do not use 'nan' here. Just output 0 (false) if there are no MTM pricing in effect.

- mtm_charge: Additional charges or premiums specifically applied when a unit operates on a month-to-month basis, separate from the base rent amount.
    - Must only exist when 'is_mtm' = 1 (true)
    - Check dedicated MTM columns, charge code breakdowns, or rent differentials
    - Must be converted to actual dollar amount if shown as percentage. e.g. 10% MTM premium on 1000 rent = 100 mtm_charge
    - If no MTM premium found, output 'nan' for mtm_charge.

- rent_market: This is the current market rent of a unit in that area.
    - Extract Market Rent from the column that provides it directly.
    - One good way to spot the Market Rent is for the same type of unit this is same most of the time.
    - Do not confuse 'rent_market' with 'rent_charge'.
    - 'rent_market' is an assumed figure by the property owner while the 'rent_charge' is the actual amount paid by the tenat.
    - Market Rent represents the potential rent for the unit based on current market conditions, not necessarily what the current tenant is paying.
    - This value should be present for both occupied and vacant units.
    - If Market Rent value for a unit is not explicitly provided, do not attempt to infer or calculate it. In such cases, report it as 'nan'.

5. Column Mapping and Inference:
- Map each piece of information to its actual column name from the data grid.
- If a standard piece of information doesn't have a corresponding column in the original data, use the standard name for both parts of the tuple in col_map.
- Do not infer or guess column names that are not present in the data grid.
- If occupancy is not explicitly provided, derive it based on Current Lease Charge as described above.
- IMPORTANT: In any case where there is no column name for a standard column, use the standard name for both the standard_column_name and actual_column_name_from_data_grid in the col_map.

6. Output Format:
- Your response should be in JSON format, following the structure shown in the <EXAMPLE_OUTPUT_JSON> </EXAMPLE_OUTPUT_JSON> section.
- Do not include any text outside of the JSON object in your response.
- Do not use any markdown formatting in your response.

- Respond with a JSON object containing "units".
- IMPORTANT: I also need the row_index column for each unit in the output.

- Follow the structure shown in the <EXAMPLE_OUTPUT_JSON> </EXAMPLE_OUTPUT_JSON> section.
- Include only the JSON object in your response, without any additional text or formatting.
- When you're using 'nan' for a certain value. ALWAYS use the string 'nan' (with quotes) instead of bare nan.

- VERY IMPORTANT: Output all units from the RR. You are not allowed to miss any units.
- IMPORTANT: Do not refrain from outputting certain units just because those units also contained in RR_FIRST_30_INPUT.
- If the RR data structure is drastically different from the structure of 'RR_FIRST_30_INPUT', then just return an empty list for 'units'.

- I'm also attaching the first 30 rows extract RR_FIRST_30_INPUT of the rent roll and your processed output RR_FIRST_30_OUTPUT for your reference.
- This will give you an understanding of how you processed the input and column headers which might be missing from the current input.
- I need you to keep the same headers exactly as the RR_FIRST_30_OUTPUT and keep the same structure.
- In case RR_FIRST_30_INPUT and RR_FIRST_30_OUTPUT are not given this is your first input.
"""

# --- Context Sections ---

CONTEXT_TEMPLATE = """
<RR>
{rr}
</RR>

<EXAMPLE_OUTPUT_JSON>
{example_output_json}
</EXAMPLE_OUTPUT_JSON>

<RR_FIRST_30_INPUT>
{rr_first_30_input}
</RR_FIRST_30_INPUT>

<RR_FIRST_30_OUTPUT>
{rr_first_30_output}
</RR_FIRST_30_OUTPUT>

<RR_PREVIOUS_OUTPUT>
{rr_prev_output}
</RR_PREVIOUS_OUTPUT>
"""

EXAMPLE_OUTPUT = {
  "units": [
    {
      "row_index": 10,
      "unit_num": "A101",
      "unit_type": "1BR/1BA",
      "sqft": 750,
      "br": 1.0,
      "bath": 1.0,
      "tenant": "John Doe",
      "move_in": "12/15/2022",
      "lease_start": "01/01/2023",
      "lease_end": "01/01/2024",
      "rent_market": 1150,
      "rent_charge": 1250,
      "rent_gov_subsidy": 0,
      "is_mtm": 1,
      "mtm_charge": 125 # Should be number if not 'nan'
    },
    {
      "row_index": 12,
      "unit_num": "A102",
      "unit_type": "Studio",
      "sqft": 500,
      "br": 0.0,
      "bath": 1.0,
      "tenant": "nan", # Use 'nan' for vacant as per instructions
      "move_in": "nan",
      "lease_start": "nan",
      "lease_end": "nan",
      "rent_market": 1110,
      "rent_charge": 0,
      "rent_gov_subsidy": 0,
      "is_mtm": 0,
      "mtm_charge": "nan"
    },
    {
      "row_index": 15,
      "unit_num": "A103",
      "unit_type": "2BR/2BA",
      "sqft": 1000,
      "br": 2.0,
      "bath": 2.0,
      "tenant": "Jane Smith",
      "move_in": "06/01/2022",
      "lease_start": "06/01/2022",
      "lease_end": "05/31/2023",
      "rent_market": 1700,
      "rent_charge": 300,
      "rent_gov_subsidy": 1500,
      "is_mtm": 0,
      "mtm_charge": "nan"
    }
  ]
}

# --- Helper Function ---

def generate_full_prompt(prompt_part, rr_data, first_30_input="", first_30_output="", prev_output=""):
    """
    Combines a prompt part with the context data.

    Args:
        prompt_part (str): Either PROMPT_PART_1_CLAUDE or PROMPT_PART_2_GEMINI.
        rr_data (str): The main rent roll data extract.
        first_30_input (str, optional): The first 30 rows input. Defaults to "".
        first_30_output (str, optional): The first 30 rows processed output. Defaults to "".
        prev_output (str, optional): The output from the previous chunk/run. Defaults to "".

    Returns:
        str: The fully formatted prompt string.
    """
    context = CONTEXT_TEMPLATE.format(
        rr=rr_data,
        example_output_json=json.dumps(EXAMPLE_OUTPUT, indent=2),
        rr_first_30_input=first_30_input if first_30_input else "N/A",
        rr_first_30_output=first_30_output if first_30_output else "N/A",
        rr_prev_output=prev_output if prev_output else "N/A"
    )
    return prompt_part + "\n" + context

# --- PDF Processing and LLM Interaction ---

def process_rent_roll_file(file_path):
    """
    Processes a rent roll file (PDF or XLSX) using Claude (for PDFs and initial analysis)
    and Gemini for detailed analysis.
    """
    if not claude_client or not gemini_model:
        print("LLM clients not initialized properly. Exiting.")
        return None

    print(f"Processing file: {file_path}")
    filename = os.path.basename(file_path)
    output_filename = os.path.splitext(filename)[0] + "_analysis.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 1. Determine File Type and Extract Text
    extracted_text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            print("File type: PDF")
            with open(file_path, "rb") as f:
                pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

            print("Sending PDF to Claude for text extraction via Portkey...")
            # Use Portkey with Claude virtual key
            portkey_claude_extraction = Portkey(
                api_key=PORTKEY_API_KEY,
                virtual_key=PORTKEY_CLAUDE_VIRTUAL_KEY
            )
            
            # Using Portkey's chat.completions interface for Claude
            message = portkey_claude_extraction.chat.completions.create(
                model="claude-3-7-sonnet-20250219",  # Using user specified model
                max_tokens=4000,  # Increased max_tokens for potentially large PDFs
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract all text content from this rent roll document accurately. Preserve the structure and formatting as much as possible, representing tables clearly.",
                            },
                        ],
                    }
                ],
            )
            # Extract text from Claude's response
            if message.content and isinstance(message.content, list):
                for block in message.content:
                    if block.type == "text":
                        extracted_text += block.text + "\n"
            extracted_text = extracted_text.strip()

            if not extracted_text:
                print("Claude returned no text content.")
                return None
            print("Text extracted successfully by Claude.")

        elif file_path.lower().endswith(".xlsx"):
            print("File type: XLSX")
            try:
                df = pd.read_excel(file_path)
                extracted_text = df.to_string()  # Convert DataFrame to string
                print("Excel data read successfully using pandas.")
            except Exception as e:
                print(f"Error reading Excel file with pandas: {e}")
                return None

        else:
            print("Unsupported file type. Only PDF and XLSX files are supported.")
            return None

    except Exception as e:
        print(f"Error during file processing: {e}")
        return None

    # 2. Claude Call: Initial Analysis with PROMPT_PART_1_CLAUDE
    try:
        # Prepare the prompt for Claude using the extracted text
        claude_prompt_text = generate_full_prompt(
            PROMPT_PART_1_CLAUDE,
            rr_data=extracted_text,
            first_30_input="N/A",  # Placeholder - implement if needed
            first_30_output="N/A",  # Placeholder - implement if needed
            prev_output="N/A",  # Placeholder - implement if needed
        )

        print("Sending extracted text to Claude for initial analysis via Portkey...")
        # Use Portkey with Claude virtual key
        portkey_claude_analysis = Portkey(
            api_key=PORTKEY_API_KEY,
            virtual_key=PORTKEY_CLAUDE_VIRTUAL_KEY
        )
        
        # Using Portkey's chat.completions interface for Claude
        claude_completion = portkey_claude_analysis.chat.completions.create(
            model="claude-3-7-sonnet-20250219",  # Using user specified model
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": claude_prompt_text
                }
            ]
        )
        
        # Extract Claude's response from Portkey's completion format
        claude_response_text = claude_completion.choices[0].message.content
        claude_response_text = claude_response_text.strip()
        
        print("Initial analysis received from Claude.")
        
        # 3. Gemini Call: Detailed Analysis with PROMPT_PART_2_GEMINI and Claude's response
        # Prepare the prompt for Gemini using the extracted text and Claude's analysis
        gemini_prompt_text = generate_full_prompt(
            PROMPT_PART_2_GEMINI,
            rr_data=extracted_text,
            first_30_input="N/A",  # Placeholder - implement if needed
            first_30_output="N/A",  # Placeholder - implement if needed
            prev_output=claude_response_text,  # Use Claude's response as previous output
        )

        print("Sending extracted text and Claude's analysis to Gemini via Portkey...")
        # Use Portkey with Gemini virtual key
        portkey_gemini = Portkey(
            api_key=PORTKEY_API_KEY,
            virtual_key=PORTKEY_GEMINI_VIRTUAL_KEY
        )
        
        # Create a completion using Portkey
        gemini_completion = portkey_gemini.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": gemini_prompt_text}]
        )
        
        # Extract the response text
        gemini_response_text = gemini_completion.choices[0].message.content
        
        # Clean potential markdown code block fences
        gemini_response_text = gemini_response_text.strip()
        if gemini_response_text.startswith("```json"):
            gemini_response_text = gemini_response_text[7:]
        if gemini_response_text.endswith("```"):
            gemini_response_text = gemini_response_text[:-3]
        gemini_response_text = gemini_response_text.strip()

        print("Final analysis received from Gemini.")

        # 4. Parse Gemini's JSON response
        try:
            analysis_result = json.loads(gemini_response_text)
            print("Gemini response parsed successfully.")
            return analysis_result
        except json.JSONDecodeError as json_e:
            print(f"Error parsing JSON response from Gemini: {json_e}")
            print("--- Gemini Raw Response causing error ---")
            print(gemini_response_text)
            print("----------------------------------------")
            return None

    except Exception as e:
        print(f"Error during API calls: {e}")
        if hasattr(e, "response"):
            print(f"API Response Error Details: {e.response}")
        return None


# --- Main Execution ---

if __name__ == "__main__":
    # Create input/output directories if they don't exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input directory: '{INPUT_DIR}'")
    print(f"Output directory: '{OUTPUT_DIR}'")

    # Find the first PDF or XLSX file in the input directory
    file_to_process = None
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".pdf", ".xlsx")):
            file_to_process = os.path.join(INPUT_DIR, filename)
            break  # Process only the first file found

    if not file_to_process:
        print(f"No PDF or XLSX files found in the '{INPUT_DIR}' directory.")
        print(
            "Please place a rent roll PDF or XLSX file in this directory and run the script again."
        )
        sys.exit(1)  # Exit if no PDF or XLSX

    # Process the found file
    analysis_data = process_rent_roll_file(file_to_process)

    # Save the result if analysis was successful
    if analysis_data:
        output_filename = os.path.splitext(os.path.basename(file_to_process))[0] + "_analysis.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            with open(output_path, "w") as f:
                json.dump(analysis_data, f, indent=2)
            print(f"Analysis successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error writing analysis JSON to file {output_path}: {e}")
    else:
        print("File processing failed. No output file generated.")

    # --- Placeholder for Portkey Integration ---
    # You would typically wrap the client initialization or the specific API calls
    # E.g., claude_client = portkey.Anthropic(...) or wrap the .create() call

    # --- Placeholder for DeepEval/Confident Integration ---
    # After getting analysis_data (if successful):
    # test_case = LLMTestCase(input=extracted_text, actual_output=json.dumps(analysis_data), ...)
    # metric.measure(test_case) ... etc.
