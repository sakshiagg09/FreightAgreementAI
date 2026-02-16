"""
SAP API Client with CSRF Token Management
Handles authentication and CSRF token caching per session
"""
import logging
import requests
from typing import Optional, Dict
from config.dev_config import (
    SAP_BASE_URL,
    SAP_USERNAME,
    SAP_PASSWORD,
    SAP_PURCHASING_ORG,
    SAP_BUSINESS_PARTNER,
    SAP_FREIGHT_AGREEMENT_BASE,
    SAP_RATE_TABLE_BASE
)

logger = logging.getLogger(__name__)

class SAPClient:
    """SAP API client with CSRF token caching per session"""
    
    # Class-level cache: (session_id, endpoint) -> csrf_token
    _csrf_token_cache: Dict[str, str] = {}
    # Class-level cache: (session_id, endpoint) -> requests.Session
    _session_cache: Dict[str, requests.Session] = {}
    
    @classmethod
    def get_csrf_token(cls, session_id: str, endpoint: str, force_refresh: bool = False) -> str:
        """
        Get CSRF token for a session. Fetches once, caches, and reuses.
        
        Args:
            session_id: Session identifier for caching
            endpoint: SAP endpoint URL (for token fetch)
            force_refresh: Force refresh token even if cached
        
        Returns:
            CSRF token string
        """
        cache_key = f"{session_id}_{endpoint}"
        
        if not force_refresh and cache_key in cls._csrf_token_cache:
            logger.debug(f"Reusing cached CSRF token for session {session_id}")
            return cls._csrf_token_cache[cache_key]
        
        # Fetch new token
        logger.info(f"Fetching CSRF token for session {session_id} (one-time per session and endpoint)")
        token = cls._fetch_csrf_token(session_id, endpoint)
        cls._csrf_token_cache[cache_key] = token
        return token
    
    @classmethod
    def _fetch_csrf_token(cls, session_id: str, endpoint: str) -> str:
        """
        Fetch CSRF token from SAP endpoint.
        Matches Postman behavior: GET to base URL (with trailing slash) with X-CSRF-Token: Fetch
        
        Args:
            session_id: Session identifier
            endpoint: Full SAP endpoint URL (e.g., .../FreightAgreement)
        
        Returns:
            CSRF token string
        """
        try:
            # Use a persistent requests.Session so that cookies (e.g. SAP_SESSIONID)
            # obtained during CSRF token fetch are automatically reused for the POST.
            session = cls._get_or_create_session(session_id, endpoint)
            
            # For CSRF token fetch, use the base URL (with trailing slash) not the full endpoint
            # Example: .../0001/ instead of .../0001/FreightAgreement
            base_url = endpoint.rsplit('/', 1)[0] + '/' if '/' in endpoint else endpoint
            if not base_url.endswith('/'):
                base_url += '/'
            
            headers = {
                "X-CSRF-Token": "Fetch",
                "Accept": "application/json",
                "Authorization": f"Basic {cls._get_basic_auth()}",  # Explicit auth header like Postman
            }
            
            logger.debug(f"Fetching CSRF token from base URL: {base_url}")
            response = session.get(
                base_url,
                headers=headers,
                timeout=(10, 30),
            )
            response.raise_for_status()
            
            csrf_token = response.headers.get("X-CSRF-Token", "")
            if not csrf_token:
                raise ValueError("CSRF token not found in response headers")
            
            # Log cookies received (for debugging)
            cookies_received = dict(response.cookies)
            if cookies_received:
                logger.debug(f"Cookies received from SAP: {list(cookies_received.keys())}")
            
            logger.info(f"CSRF token fetched successfully: {csrf_token[:20]}...")
            return csrf_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch CSRF token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}, body: {e.response.text[:500]}")
                logger.error(f"Response headers: {dict(e.response.headers)}")
            raise Exception(f"CSRF token fetch failed: {str(e)}")
    
    @classmethod
    def get_auth_headers(cls, session_id: str, endpoint: str, content_type: str = "application/json") -> dict:
        """
        Get authentication headers with CSRF token.
        
        Args:
            session_id: Session identifier
            endpoint: SAP endpoint URL
            content_type: Content-Type header value
        
        Returns:
            Dictionary of headers
        """
        """
        Backwards-compatible helper that only returns headers.
        NOTE: Prefer using `post_with_csrf` so that cookies are reused correctly.
        """
        csrf_token = cls.get_csrf_token(session_id, endpoint)
        
        return {
            "X-CSRF-Token": csrf_token,
            "Accept": "application/json",
            "Content-Type": content_type,
            "Authorization": f"Basic {cls._get_basic_auth()}",
        }
    
    @classmethod
    def _get_basic_auth(cls) -> str:
        """Get Basic Auth header value"""
        import base64
        credentials = f"{SAP_USERNAME}:{SAP_PASSWORD}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return encoded
    
    @classmethod
    def _get_or_create_session(cls, session_id: str, endpoint: str) -> requests.Session:
        """
        Get or create a persistent requests.Session for a given (session_id, endpoint).
        This ensures that cookies set during CSRF token fetch (e.g. SAP_SESSIONID)
        are reused for subsequent POST calls, matching Postman/browser behavior.
        """
        cache_key = f"{session_id}_{endpoint}"
        if cache_key not in cls._session_cache:
            session = requests.Session()
            # Set basic auth on the session so all requests use it.
            session.auth = (SAP_USERNAME, SAP_PASSWORD)
            cls._session_cache[cache_key] = session
        return cls._session_cache[cache_key]
    
    @classmethod
    def post_with_csrf(
        cls,
        session_id: str,
        endpoint: str,
        json: dict,
        content_type: str = "application/json",
        timeout: tuple = (10, 60),
    ) -> requests.Response:
        """
        Perform a POST request with CSRF token and cookie reuse.
        This matches the behavior of your working Postman request where:
        - A CSRF token is fetched first
        - Session cookies (SAP_SESSIONID, sap-usercontext, etc.) are reused
        """
        session = cls._get_or_create_session(session_id, endpoint)
        csrf_token = cls.get_csrf_token(session_id, endpoint)
        
        headers = {
            "X-CSRF-Token": csrf_token,
            "Accept": "application/json",
            "Content-Type": content_type,
            "Authorization": f"Basic {cls._get_basic_auth()}",
        }
        
        # Log cookies being sent (for debugging)
        cookies_to_send = dict(session.cookies)
        if cookies_to_send:
            logger.debug(f"Sending cookies with POST: {list(cookies_to_send.keys())}")
        
        logger.debug(f"POST to {endpoint} with CSRF token: {csrf_token[:20]}...")
        response = session.post(
            endpoint,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        logger.debug(
            f"POST response: status={response.status_code} content_length={len(response.content) if response.content else 0}",
            extra={"status_code": response.status_code, "url": endpoint},
        )
        return response

    @classmethod
    def get_with_session(
        cls,
        session_id: str,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
        timeout: tuple = (10, 60),
        session_base: Optional[str] = None,
    ) -> requests.Response:
        """
        Perform a GET request reusing the session (cookies, auth) from CSRF token fetch.
        Used for reading Freight Agreement data with expand/filter.

        When session_base is provided, use it for session/CSRF caching (not endpoint).
        This ensures GET and POST share the same cookies when hitting different URLs
        of the same service (e.g. rate table GET vs $batch POST).
        """
        cache_key = session_base if session_base is not None else endpoint
        session = cls._get_or_create_session(session_id, cache_key)
        cls.get_csrf_token(session_id, cache_key)
        headers = {
            "X-CSRF-Token": cls._csrf_token_cache.get(f"{session_id}_{cache_key}", ""),
            "Accept": "application/json",
            "Authorization": f"Basic {cls._get_basic_auth()}",
        }
        logger.debug(f"GET {endpoint} params={params}")
        response = session.get(endpoint, headers=headers, params=params, timeout=timeout)
        logger.debug(
            f"GET response: status={response.status_code} content_length={len(response.content) if response.content else 0}",
            extra={"status_code": response.status_code, "url": endpoint},
        )
        return response

    @classmethod
    def post_with_csrf_raw(
        cls,
        session_id: str,
        endpoint: str,
        data: str,
        content_type: str,
        timeout: tuple = (10, 600),
        session_base: Optional[str] = None,
    ) -> requests.Response:
        """
        Perform a POST request with raw body (e.g. multipart/mixed for $batch).
        Reuses CSRF token and session cookies like post_with_csrf.

        When session_base is provided, use it for session/CSRF caching (not endpoint).
        This ensures GET and POST share the same cookies when hitting different URLs
        of the same service (e.g. rate table GET vs $batch POST).
        """
        cache_key = session_base if session_base is not None else endpoint
        session = cls._get_or_create_session(session_id, cache_key)
        csrf_token = cls.get_csrf_token(session_id, cache_key)
        headers = {
            "X-CSRF-Token": csrf_token,
            "Accept": "multipart/mixed",
            "Content-Type": content_type,
            "Authorization": f"Basic {cls._get_basic_auth()}",
        }
        body_size = len(data) if isinstance(data, (str, bytes)) else 0
        logger.info(
            f"SAP $batch POST: url={endpoint} body_size={body_size} timeout={timeout}",
            extra={"url": endpoint, "body_size_bytes": body_size, "session_id": session_id},
        )
        response = session.post(endpoint, headers=headers, data=data, timeout=timeout)
        logger.info(
            f"SAP $batch POST response: status={response.status_code} content_length={len(response.content) if response.content else 0}",
            extra={"status_code": response.status_code, "url": endpoint, "response_size_bytes": len(response.content) if response.content else 0},
        )
        if response.status_code not in (200, 201, 202):
            logger.warning(
                f"SAP $batch POST non-success: status={response.status_code} body_preview={str(response.text[:300]) if response.text else 'N/A'}...",
                extra={"status_code": response.status_code, "url": endpoint},
            )
        return response

    @classmethod
    def clear_cache(cls, session_id: Optional[str] = None):
        """
        Clear CSRF token cache.
        
        Args:
            session_id: If provided, clear only this session's cache. Otherwise clear all.
        """
        if session_id:
            # Clear CSRF tokens and sessions for this session_id
            csrf_keys_to_remove = [k for k in cls._csrf_token_cache.keys() if k.startswith(session_id)]
            for key in csrf_keys_to_remove:
                del cls._csrf_token_cache[key]
            session_keys_to_remove = [k for k in cls._session_cache.keys() if k.startswith(session_id)]
            for key in session_keys_to_remove:
                sess = cls._session_cache.pop(key, None)
                if sess:
                    sess.close()
            logger.info(f"Cleared CSRF token and session cache for session {session_id}")
        else:
            cls._csrf_token_cache.clear()
            # Close and clear all sessions
            for sess in cls._session_cache.values():
                try:
                    sess.close()
                except Exception:
                    pass
            cls._session_cache.clear()
            logger.info("Cleared all CSRF token and session caches")
