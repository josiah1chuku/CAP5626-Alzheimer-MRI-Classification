# Security Policy

## Project Context

This is an academic project for CAP5626 at Florida A&M University.
The codebase implements deep learning for Alzheimer's disease
MRI classification.

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | YES       |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do not** open a public GitHub issue
2. Email: josiah1.chuku@famu.edu
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact

## Security Measures Implemented

- GitHub Secret Scanning enabled
- Dependabot dependency alerts enabled
- Automated pip-audit scanning on every push
- Bandit code security scanning
- detect-secrets pre-commit scanning
- No credentials stored in code (using Colab Secrets)
- No sensitive patient data stored in repository
- ADNI dataset accessed via institutional credentials only

## Dependencies

All dependencies are pinned in requirements.txt.
Run pip-audit to check for vulnerabilities:

    pip install pip-audit
    pip-audit -r requirements.txt
