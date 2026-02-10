## Framework Comparison: Pros and Cons For SharedHands

#### Flask (Python) :

Pros

* Strong compatibility with Python ML libraries such as TensorFlow, PyTorch, Mediapipe for ASl tools.
* Great choice for prototyping and small to medium-sized ASL systems
* Flexible and highly customizable
* Minimal framework overhead which would end up being faster development and debugging process

Cons

* Limited built-in asynchronous support
* Requires additional extensions for scalability and security
* Manual setup needed for larger or more complex projects
* Performance optimization requires extra configuration

#### FastAPI (Python) :

Pros

* Excellent performance due to async support and modern architecture
* Native integration with Python ML libraries
* Strong typing and data validation using Pydantic
* Easy deployment with Uvicorn and Docker
* Automatic API documentation (Swagger / OpenAPI)

Cons

* Requires some sort of learning curve for some of the advanced features
* Newer framework with a smaller ecosystem than Django
* Debugging async ML pipelines can be complex

#### Django (Python):

Pros

* Strong Python ML compatibility
* Good for large-scale applications with complex data models
* Full-featured framework with built-in authentication, ORM, and admin panel
* Mature, stable, and widely adopted

Cons

* Heavier and more complex than necessary for API-only servers
* Slower development for simple APIs due to boilerplate
* Async support exists but is not as streamlined as FastAPI
* Can introduce unnecessary overhead for ML inference API’s

#### Express.js (Node.js):

Pros

* Fast and lightweight
* Excellent asynchronously handling via event-driven architecture
* Large ecosystems and community support
* High performance for general-purpose APIs

Cons

* Not as compatible for ML-heavy backend systems
* Increases system complexity when used for ML workloads
* Requires separate Python services for ML integration
* Ml inference in Node.js is limited compared to python

## Team Discussion:

We as a team have discussed the options that I’ve research in order to determine which framesworks would be compatible in support of API development, performance, scalability and simplity. We decided to choose Flask due to its simplicity, flexibility and its strong integration with Python’s based machine learning libraries. The backend’s responsibility is to support ML-driven functionality rather than manage complex application logic, thus Flask provides an efficient and easier solution without unnecessary complications.

Futhermore, Flasks allows the development team to maintain control over the structure of the application which will remain easy as our project requirements continue to evolve. With minimal design, it allows for rapid development, prototyping and incremental scaling with ease.