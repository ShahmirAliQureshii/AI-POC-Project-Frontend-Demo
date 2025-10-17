# Face Recognition Next.js Site

This project is a web application built using Next.js that serves as a frontend for a face recognition system. It provides a user-friendly interface to interact with the existing Python backend application, showcasing various features, model management, and real-time analytics.

## Project Structure

The project is organized as follows:

```
face-recognition-nextjs-site
├── app
│   ├── layout.js                # Main layout with header and footer
│   ├── globals.css              # Global styles for the application
│   ├── page.js                  # Home page showcasing features
│   ├── features
│   │   └── page.js              # Features of the face recognition system
│   ├── model
│   │   └── page.js              # Model viewer and uploader
│   ├── dashboard
│   │   └── page.js              # Live dashboard displaying real-time data
│   ├── contact
│   │   └── page.js              # Contact/About page
│   └── api
│       └── recognition
│           └── route.js         # API proxy to Python backend
├── components
│   ├── Header.js                # Header component with navigation
│   ├── Footer.js                # Footer component with copyright info
│   ├── Hero.js                  # Hero section for the home page
│   ├── FeatureCard.js           # Card component for features
│   ├── ModelCard.js             # Card component for models
│   ├── DashboardGrid.js          # Grid layout for the dashboard
│   └── ContactForm.js           # Contact form component
├── lib
│   ├── api.js                   # Client helpers for API calls
│   └── utils.js                 # Utility functions
├── styles
│   ├── tailwind.css             # Tailwind CSS styles
│   └── theme.css                # Custom theme styles
├── public
│   └── robots.txt               # Instructions for web crawlers
├── scripts
│   └── start-python-backend.sh  # Script to run the Python backend
├── .gitignore                   # Files to ignore by Git
├── package.json                 # Project metadata and dependencies
├── next.config.js               # Next.js configuration
├── postcss.config.js            # PostCSS configuration
├── tailwind.config.js           # Tailwind CSS configuration
├── jsconfig.json                # JavaScript project configuration
└── README.md                    # Project documentation
```

## Features

- **Home Page**: Introduces the face recognition system and its capabilities.
- **Features Page**: Details the features and advantages of the system.
- **Model Management**: Allows users to upload and view trained models.
- **Live Dashboard**: Displays real-time analytics and data from the face recognition system.
- **Contact Page**: Provides information about the project and a contact form for inquiries.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd face-recognition-nextjs-site
   ```

2. **Install dependencies**:
   ```
   npm install
   ```

3. **Run the development server**:
   ```
   npm run dev
   ```

4. **Start the Python backend** (optional):
   You can run the existing Python application using the provided script:
   ```
   ./scripts/start-python-backend.sh
   ```

5. **Open your browser**:
   Navigate to `http://localhost:3000` to view the application.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.