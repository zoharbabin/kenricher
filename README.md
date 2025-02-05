# Kaltura Metadata Generator

*Because your videos deserve more than just captions—give them a metadata makeover!*

Kenricher is an open-source tool that leverages AI to automatically enrich and optimize Kaltura media metadata, enhancing video discoverability and engagement. Designed for content creators and developers, it streamlines metadata management to help you achieve a polished, professional digital presence.

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Enhancements & Contributing](#enhancements--contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About

The **Kaltura Metadata Generator** is an open-source Python project designed to process video/audio entries from a Kaltura account, extract captions, and generate snappy metadata using OpenAI’s powerful language models. Think of it as a digital stylist that gives your videos a fresh title, a clever description, and a neat set of tags—all while keeping things colorful (both literally and figuratively) with our logging and UI enhancements.

If you’re tired of bland metadata and want to spice things up with AI magic, you’ve come to the right place. And hey, if our code were any cooler, it’d be wearing sunglasses indoors!

---

## Features

- **Automated Caption Processing:** Splits long transcripts into manageable chunks and summarizes each with finesse.
- **AI-Powered Metadata Generation:** Uses OpenAI’s GPT models to generate titles, tags, and descriptions based on content.
- **Kaltura Integration:** Fetches media entries and updates metadata via the Kaltura API.
- **Rich Console UI:** Provides real-time feedback with progress spinners, tables, and color-coded logging.
- **Retry Decorators & Custom Logging:** Ensures robust error-handling and smooth execution.
- **Batch Processing:** Process multiple entries by category ID or specific entry IDs.
- **Auto-Update Mode:** Option to automatically update entries without manual confirmation.
- **Simulation Mode:** Test drive metadata generation without updating your live entries.

---

## Prerequisites

- **Python:** 3.7+

### Required Python Packages

- `coloredlogs`
- `KalturaClient`
- `python-dotenv`
- `rich`
- `requests`
- `pydantic`
- `openai`

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/kaltura-metadata-generator.git
   cd kaltura-metadata-generator
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install coloredlogs KalturaClient python-dotenv rich requests pydantic openai
   ```

---

## Configuration

This project relies on several environment variables to connect to your Kaltura account and OpenAI API. Create a `.env` file in the root directory with the following:

```ini
# Kaltura Configuration
KALTURA_PARTNER_ID=<your_kaltura_partner_id>
KALTURA_ADMIN_SECRET=<your_kaltura_admin_secret>
KALTURA_SERVICE_URL=<your_kaltura_service_url>

# OpenAI Configuration
OPENAI_API_KEY=<your_openai_api_key>
```

Replace `<your_kaltura_partner_id>`, `<your_kaltura_admin_secret>`, `<your_kaltura_service_url>`, and `<your_openai_api_key>` with your actual credentials. No one likes a secret spilled in public—keep these safe!

---

## Usage

### Running the Script

To process all video/audio entries:

```bash
python kenricher.py
```

### Process Specific Entries

If you want to process specific entries, use the `--entry_ids` flag:

```bash
python kenricher.py --entry_ids <ENTRY_ID1,ENTRY_ID2,...>
```

### Process Entries by Category

To process all entries within specific categories:

```bash
python kenricher.py --category_ids <CATEGORY_ID1,CATEGORY_ID2,...>
```

### Simulation Mode

Want to test the metadata generation without actually updating entries? Use the simulation mode:

```bash
python kenricher.py --simulate
```

### Auto-Update Mode

To automatically update entries without manual confirmation:

```bash
python kenricher.py --auto-update
```

### Additional Options

- **Debug Mode:** Enable verbose logging
  ```bash
  python kenricher.py --debug
  ```
- **Session Privileges:** Specify custom Kaltura session privileges
  ```bash
  python kenricher.py --privileges "disableentitlement"
  ```
---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as long as you include the original license and attribution.

---

*Now go on—generate metadata like a boss and let your videos shine!*

