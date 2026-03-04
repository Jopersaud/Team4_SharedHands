
# Coding Standards for SharedHands ASL Translation 
**Project Team 4 
Joshua Persaud, Antonio Datasets, Dillon, Team Members**



**Introduction**  
This document provides coding standards and best practices for the SharedHands ASL Translation project developed using Python (Flask backend), JavaScript (React frontend), and Firebase.

**General Principles** 
• Code should be easily understandable
• The coding principles defined in this document must be held across the entire project
• Code should be well-commented and documented such that any viewer can understand what a function/class/file is doing.
• Security: Never commit API keys, always validate input, and follow secure coding practices.

**Formatting and Naming Conventions**

Python Backend (Flask, Services, ML Models), JavaScript Frontend (React Components, Services):
• Use 1 blank line between methods in a class.

Naming 
• Files and Modules: Use lowercase_with_underscores.py 
	Example: video_processing_service.py 
• Classes: Use PascalCase. 
	Example: VideoProcessingService 
• Functions and Methods: Use lowercase_with_underscores starting with a verb. 		
	Example: process_video_frame(), create_user() 
• Variables: Use lowercase_with_underscores.
	Example: subscription_tier, user_email 
• Constants: Use ALL_CAPS with underscores. 
	Example: MAX_FILE_SIZE, DEFAULT_FPS 

String Quotes 
• Use single quotes 'text' for regular strings. 
Line Length 
• Lines should not exceed 100 characters.

Imports 
• Group imports in three sections:
Standard library imports
Third-party imports
Local application imports 
Sort alphabetically within each group.

Semicolons (js) • Always use semicolons at the end of statements.

Firebase and Firestore Collection Naming 
• Use camelCase plural nouns. 
Example: users, subscriptions, translations

Document Field Naming 
• Use camelCase for all field names.
Example: subscriptionTier, emailVerified, createdAt

Code Structure

Python Backend, JavaScript Frontend (React) 
• Keep functions/methods focused on a single task.
• Limit function parameters to a maximum of 5. Use dictionaries for more complex parameter sets. 
• Follow the Single Responsibility Principle for classes. 
• Group related methods together for readability.

Error Handling 
• Prefer exceptions to return codes for error handling. 
• Catch specific exceptions, not generic Exception. 
• Catch exceptions at the highest level that can properly handle them.

**Comments and Documentation**

Python 
• Use triple quotes """ for docstrings on all public functions and classes. 
• Include: brief description, Args, Returns, Raises in docstrings. 
• Inline comments should explain why a function performs its task the way it does.
• Use # TODO(name): description format for TODO comments. 

JavaScript 
• Use JSDoc comments for components and complex functions.  `/**` and end with `*/`
• Inline comments should explain complex logic or non-obvious decisions. 

**Version Control**

Branch Naming 
• Use format: type/description-in-kebab-case 
• Types: feature, bugfix or docs 
• Examples: feature/user-authentication, bugfix/video-timeout

Commit Messages 
• Use format: type: Description in present tense 
• Can be long but keep concise
• Examples: feat: Add user registration, fix: Resolve timeout

Pull Requests 
• Commit related changes in a logical unit and include a meaningful commit message. 
• Use branches for features, bug fixes, and releases. 
• All code must be reviewed by at least one other team member before merging. 

Testing 
• Write unit tests for all new classes and methods where practical. 
• Test file naming: o Python: test_<module_name>.py o JavaScript: <ComponentName>.test.js 
• Test function naming: Use descriptive names that explain what is being tested. 
 Example: test_create_user_with_valid_email_succeeds() 

Security 
• Adhere to best practices for secure coding to protect data and prevent vulnerabilities. 
• Never commit API keys, passwords, tokens, or any credentials to the repository. 
• Use environment variables (.env files) for all secrets and ensure .env is in .gitignore. 
• Use Firebase Authentication for all password operations
• Always validate and sanitize user input before processing. 
• Implement proper access controls in Firestore security rules.

Naming Best Practices Summary Names to Use 
• Descriptive names that explain what the variable represents (subscription_tier not tier). 
• Boolean names that imply true/false (is_authenticated, has_access). 
• Function names that start with verbs (create_user(), validate_input()). 
• Consistent opposites for related concepts (start/stop, begin/end, min/max).

Conclusion:
These coding standards are **mandatory** for all SharedHands contributors. Consistent application of these standards will:
• Improve code quality and readability
• Reduce bugs and debugging time
• Facilitate code reviews
• Enable faster onboarding of new team members
• Create a professional, maintainable codebase
