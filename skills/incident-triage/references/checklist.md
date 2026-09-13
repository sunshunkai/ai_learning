# Multi-Service Incident Checklist

1. Confirm the first failing request boundary.
2. Compare error rates by service, region, and release version.
3. Check whether the failure follows a deployment, configuration change, or
   traffic shift.
4. Verify dependency health before changing application code.
5. Record the rollback or feature-flag mitigation separately from the fix.
