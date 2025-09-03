# Personal Cloud Storage with AI Organization

A self-hosted **personal cloud storage system** that combines **secure file management** with **AI-powered file organization**.  
This project demonstrates practical backend development, asynchronous job processing, and AI integration with a modern **React dashboard** frontend.

## 🚀 Features

- **File Management**
  - Upload, download, delete, and rename files
  - Create and manage folders
  - Large file support with **multipart uploads**
  - Metadata storage in database (fileName, path, size, type, owner, AI tags, permissions)

- **AI Integration**
  - Auto-tagging of images using AI (e.g., Google Cloud Vision / OpenAI API)
  - AI-generated summaries for text/PDF files
  - Intelligent metadata stored alongside files

- **Asynchronous Job Processing**
  - Upload triggers an AI processing **job**
  - Background worker analyzes files & updates database
  - Non-blocking, scalable architecture using **Redis** + **BullMQ (Node.js)** or **Celery (Python)**

- **Secure File Handling**
  - Role-based access & permissions
  - Temporary secure share links
  - Encrypted storage paths

- **Modern Frontend (React)**
  - File & folder explorer dashboard
  - File upload with progress bar
  - AI tags & summaries display
  - Rename, delete, and share files directly

## How It Works 💡

Before:

```
/home/user/messy_documents/
├── IMG_20230515_140322.jpg
├── IMG_20230516_083045.jpg
├── IMG_20230517_192130.jpg
├── budget_2023.xlsx
├── meeting_notes_05152023.txt
├── project_proposal_draft.docx
├── random_thoughts.txt
├── recipe_chocolate_cake.pdf
├── scan0001.pdf
├── vacation_itinerary.docx
└── work_presentation.pptx

0 directories, 11 files
```

After:

```
/home/user/organized_documents/
├── Financial
│   └── 2023_Budget_Spreadsheet.xlsx
├── Food_and_Recipes
│   └── Chocolate_Cake_Recipe.pdf
├── Meetings_and_Notes
│   └── Team_Meeting_Notes_May_15_2023.txt
├── Personal
│   └── Random_Thoughts_and_Ideas.txt
├── Photos
│   ├── Cityscape_Sunset_May_17_2023.jpg
│   ├── Morning_Coffee_Shop_May_16_2023.jpg
│   └── Office_Team_Lunch_May_15_2023.jpg
├── Travel
│   └── Summer_Vacation_Itinerary_2023.docx
└── Work
    ├── Project_X_Proposal_Draft.docx
    ├── Quarterly_Sales_Report.pdf
    └── Marketing_Strategy_Presentation.pptx

7 directories, 11 files
```

## Roadmap 📅

- [ ] Copilot Mode: chat with AI to tell AI how you want to sort the file (ie. read and rename all the PDFs)
- [ ] Change models with CLI 
- [ ] ebook format support
- [ ] audio file support
- [ ] video file support
- [ ] Implement best practices like Johnny Decimal
- [ ] Check file duplication
- [ ] Dockerfile for easier installation
- [ ] People from [Nexa](https://github.com/NexaAI/nexa-sdk) is helping me to make executables for macOS, Linux and Windows


## Supported File Types 📁

- **Images:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- **Text Files:** `.txt`, `.docx`, `.md`
- **Spreadsheets:** `.xlsx`, `.csv`
- **Presentations:** `.ppt`, `.pptx`
- **PDFs:** `.pdf`

## Prerequisites 💻

- **Operating System:** Compatible with Windows, macOS, and Linux.
- **Python Version:** Python 3.12
- **Conda:** Anaconda or Miniconda installed.
- **Git:** For cloning the repository (or you can download the code as a ZIP file).



