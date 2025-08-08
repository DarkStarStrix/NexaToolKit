# Security Policy

This document outlines the security measures implemented in the Nexa ToolKit backend to protect user data and ensure service reliability.

## Security Measures

-   **Secrets Management**: All sensitive information, including Stripe API keys and webhook secrets, is managed via environment variables and is never hardcoded. We provide a `.env.example` file for local development.

-   **Input Validation**: The FastAPI framework provides automatic data validation for all incoming request bodies based on Pydantic models. This prevents common data-based vulnerabilities.

-   **Rate Limiting**: To prevent abuse and DoS attacks, API endpoints are rate-limited using `slowapi`. This applies to computationally intensive or sensitive endpoints like checkout creation and webhook processing.

-   **CORS**: Cross-Origin Resource Sharing (CORS) middleware is configured to restrict access. In a production environment, this should be locked down to the specific domain of the frontend application.

-   **Error Handling**: A global exception handler is in place to catch all unhandled errors. It prevents leaking sensitive stack traces or configuration details to the client, returning a generic 500 error instead.

-   **Dependency Management**: Dependencies are managed via `requirements.txt`. It is recommended to use tools like `pip-audit` or GitHub's Dependabot to regularly scan for vulnerabilities in dependencies.

-   **HTTPS**: The application is designed to be deployed behind a reverse proxy (like Nginx or Traefik) that terminates SSL/TLS, ensuring all traffic is encrypted via HTTPS.

-   **Stripe Webhook Verification**: All incoming webhooks from Stripe are verified using the signature provided in the `Stripe-Signature` header to ensure they originate from Stripe and have not been tampered with.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it to us at `security@example.com`. We appreciate your efforts to disclose your findings responsibly.

