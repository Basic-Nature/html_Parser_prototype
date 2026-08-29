"""Public-only Ballot Lens browser egress guard.

This module is inert until a later runtime-wiring milestone. It provides the
request-budget, URL, DNS, redirect, and subresource decisions required by the
frozen BL-P2C public network policy. The guard accepts only a server-resolved
approved target. It never accepts a caller-supplied URL as authority.

No browser is launched and no route is installed merely by importing this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import socket
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .public_ballot_lens_policy import (
    DEFAULT_PUBLIC_RUN_POLICY,
    PublicBallotLensRunPolicy,
)


class PublicBrowserEgressError(RuntimeError):
    pass


DnsResolver = Callable[[str], Iterable[str]]

PUBLIC_BLOCKED_RESOURCE_TYPES = frozenset(
    {
        "image",
        "media",
        "font",
    }
)


@dataclass(frozen=True)
class PublicBrowserEgressDecision:
    allowed: bool
    reason: str
    request_number: int
    resource_type: str
    top_level_navigation: bool
    destination_host_sha256: str | None


def _host_hash(host: str) -> str:
    return hashlib.sha256(host.encode("utf-8")).hexdigest()


def _normalize_hostname(host: str) -> str:
    value = str(host or "").strip().rstrip(".").lower()
    if not value:
        raise PublicBrowserEgressError("Public browser URL has no host.")
    try:
        value = value.encode("idna").decode("ascii")
    except Exception as exc:
        raise PublicBrowserEgressError(
            "Public browser hostname is not valid IDNA."
        ) from exc
    return value


def _normalize_public_url(url: str) -> tuple[str, SplitResult]:
    if not isinstance(url, str) or not url.strip():
        raise PublicBrowserEgressError("Public browser URL is empty.")

    try:
        parsed = urlsplit(url.strip())
    except Exception as exc:
        raise PublicBrowserEgressError(
            "Public browser URL could not be parsed."
        ) from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise PublicBrowserEgressError(
            "Public browser URL scheme is not allowed."
        )
    if parsed.username is not None or parsed.password is not None:
        raise PublicBrowserEgressError(
            "Public browser URL userinfo is not allowed."
        )

    host = _normalize_hostname(parsed.hostname or "")

    try:
        port = parsed.port
    except ValueError as exc:
        raise PublicBrowserEgressError(
            "Public browser URL port is invalid."
        ) from exc

    if port is not None:
        allowed_port = (
            (scheme == "https" and port == 443)
            or (scheme == "http" and port == 80)
        )
        if not allowed_port:
            raise PublicBrowserEgressError(
                "Public browser URL uses a nonstandard port."
            )

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise PublicBrowserEgressError(
            "Public browser target must use a DNS hostname, not an IP literal."
        )

    default_port = 443 if scheme == "https" else 80
    netloc = host if port in (None, default_port) else f"{host}:{port}"
    normalized = urlunsplit(
        (
            scheme,
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    return normalized, urlsplit(normalized)


def default_dns_resolver(host: str) -> tuple[str, ...]:
    addresses: list[str] = []
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as exc:
        raise PublicBrowserEgressError(
            "Public browser DNS resolution failed."
        ) from exc

    for info in infos:
        try:
            address = str(info[4][0]).strip()
        except Exception:
            continue
        if address and address not in addresses:
            addresses.append(address)

    if not addresses:
        raise PublicBrowserEgressError(
            "Public browser DNS resolution returned no addresses."
        )
    return tuple(addresses)


def _validate_resolved_addresses(
    host: str,
    *,
    resolver: DnsResolver,
) -> tuple[str, ...]:
    try:
        addresses = tuple(
            str(value).strip()
            for value in resolver(host)
            if str(value).strip()
        )
    except PublicBrowserEgressError:
        raise
    except Exception as exc:
        raise PublicBrowserEgressError(
            "Public browser DNS resolver failed."
        ) from exc

    if not addresses:
        raise PublicBrowserEgressError(
            "Public browser DNS resolution returned no addresses."
        )

    checked: list[str] = []
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise PublicBrowserEgressError(
                "Public browser DNS returned an invalid IP address."
            ) from exc

        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise PublicBrowserEgressError(
                "Public browser DNS resolved to a blocked network range."
            )
        if address not in checked:
            checked.append(address)

    if not checked:
        raise PublicBrowserEgressError(
            "Public browser DNS resolution produced no usable address."
        )
    return tuple(checked)


class PublicBrowserEgressGuard:
    """Run-scoped public browser policy with per-request DNS revalidation."""

    def __init__(
        self,
        *,
        approved_target_url: str,
        policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
        resolver: DnsResolver = default_dns_resolver,
    ) -> None:
        normalized, parsed = _normalize_public_url(approved_target_url)
        self._approved_target_url = normalized
        self._approved_host = _normalize_hostname(parsed.hostname or "")
        self._policy = policy
        self._resolver = resolver
        self._request_count = 0
        self._top_level_redirect_count = 0
        self._top_level_seen = False

        # The approved server-resolved target itself must be safe before a
        # browser context can be given this guard.
        _validate_resolved_addresses(
            self._approved_host,
            resolver=self._resolver,
        )

    @property
    def request_count(self) -> int:
        return int(self._request_count)

    @property
    def top_level_redirect_count(self) -> int:
        return int(self._top_level_redirect_count)

    def validate_initial_target(self, target_url: str) -> None:
        normalized, parsed = _normalize_public_url(target_url)
        host = _normalize_hostname(parsed.hostname or "")
        if normalized != self._approved_target_url:
            raise PublicBrowserEgressError(
                "Initial public browser target differs from the "
                "server-approved registry target."
            )
        if host != self._approved_host:
            raise PublicBrowserEgressError(
                "Initial public browser host differs from approved authority."
            )
        _validate_resolved_addresses(
            host,
            resolver=self._resolver,
        )

    def evaluate_request(
        self,
        *,
        url: str,
        resource_type: str,
        top_level_navigation: bool,
    ) -> PublicBrowserEgressDecision:
        self._request_count += 1
        request_number = self._request_count

        if request_number > int(
            self._policy.browser_network_request_max
        ):
            return PublicBrowserEgressDecision(
                allowed=False,
                reason="network_request_budget_exceeded",
                request_number=request_number,
                resource_type=str(resource_type or ""),
                top_level_navigation=bool(top_level_navigation),
                destination_host_sha256=None,
            )

        resource = str(resource_type or "").strip().lower()
        if resource in PUBLIC_BLOCKED_RESOURCE_TYPES:
            return PublicBrowserEgressDecision(
                allowed=False,
                reason="resource_type_blocked",
                request_number=request_number,
                resource_type=resource,
                top_level_navigation=bool(top_level_navigation),
                destination_host_sha256=None,
            )

        try:
            normalized, parsed = _normalize_public_url(url)
            host = _normalize_hostname(parsed.hostname or "")
            _validate_resolved_addresses(
                host,
                resolver=self._resolver,
            )
        except PublicBrowserEgressError as exc:
            return PublicBrowserEgressDecision(
                allowed=False,
                reason="destination_blocked:" + str(exc),
                request_number=request_number,
                resource_type=resource,
                top_level_navigation=bool(top_level_navigation),
                destination_host_sha256=None,
            )

        if top_level_navigation:
            if host != self._approved_host:
                return PublicBrowserEgressDecision(
                    allowed=False,
                    reason="top_level_cross_host_blocked",
                    request_number=request_number,
                    resource_type=resource,
                    top_level_navigation=True,
                    destination_host_sha256=_host_hash(host),
                )

            if not self._top_level_seen:
                if normalized != self._approved_target_url:
                    return PublicBrowserEgressDecision(
                        allowed=False,
                        reason="initial_target_not_exact_approved_target",
                        request_number=request_number,
                        resource_type=resource,
                        top_level_navigation=True,
                        destination_host_sha256=_host_hash(host),
                    )
                self._top_level_seen = True
            elif normalized != self._approved_target_url:
                self._top_level_redirect_count += 1
                if self._top_level_redirect_count > int(
                    self._policy.top_level_redirect_max
                ):
                    return PublicBrowserEgressDecision(
                        allowed=False,
                        reason="top_level_redirect_limit_exceeded",
                        request_number=request_number,
                        resource_type=resource,
                        top_level_navigation=True,
                        destination_host_sha256=_host_hash(host),
                    )

        return PublicBrowserEgressDecision(
            allowed=True,
            reason="allowed",
            request_number=request_number,
            resource_type=resource,
            top_level_navigation=bool(top_level_navigation),
            destination_host_sha256=_host_hash(host),
        )

    def decide_playwright_request(
        self,
        request: Any,
    ) -> PublicBrowserEgressDecision:
        url = getattr(request, "url", "")
        resource_type = getattr(request, "resource_type", "")
        nav_attr = getattr(request, "is_navigation_request", None)

        try:
            top_level_navigation = (
                bool(nav_attr())
                if callable(nav_attr)
                else bool(nav_attr)
            )
        except Exception:
            top_level_navigation = False

        return self.evaluate_request(
            url=str(url or ""),
            resource_type=str(resource_type or ""),
            top_level_navigation=top_level_navigation,
        )

    def handle_sync_route(self, route: Any, request: Any) -> None:
        decision = self.decide_playwright_request(request)
        if decision.allowed:
            route.continue_()
        else:
            route.abort("blockedbyclient")

    async def handle_async_route(
        self,
        route: Any,
        request: Any,
    ) -> None:
        decision = self.decide_playwright_request(request)
        if decision.allowed:
            await route.continue_()
        else:
            await route.abort("blockedbyclient")
