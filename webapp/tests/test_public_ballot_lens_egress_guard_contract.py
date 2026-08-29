from __future__ import annotations

from dataclasses import replace

import pytest

from webapp.parser.services.public_ballot_lens_egress import (
    PUBLIC_BLOCKED_RESOURCE_TYPES,
    PublicBrowserEgressError,
    PublicBrowserEgressGuard,
)
from webapp.parser.services.public_ballot_lens_policy import (
    DEFAULT_PUBLIC_RUN_POLICY,
)


APPROVED = "https://results.example.gov/elections/2024"


class RecordingResolver:
    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.calls = []

    def __call__(self, host):
        self.calls.append(host)
        return self.mapping.get(host, ())


def public_resolver():
    return RecordingResolver(
        {
            "results.example.gov": ("203.0.113.10",),
            "cdn.example.net": ("198.51.100.20",),
        }
    )


# Python's documentation-only TEST-NET ranges are not universally classified
# as globally routable by ipaddress, so tests use known public resolver answers.
PUBLIC_V4 = "8.8.8.8"
PUBLIC_V4_2 = "1.1.1.1"


def resolver_with_public_hosts():
    return RecordingResolver(
        {
            "results.example.gov": (PUBLIC_V4,),
            "cdn.example.net": (PUBLIC_V4_2,),
        }
    )


def test_guard_requires_exact_server_approved_initial_target():
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=resolver,
    )
    guard.validate_initial_target(APPROVED)

    with pytest.raises(PublicBrowserEgressError):
        guard.validate_initial_target(
            "https://results.example.gov/elections/other"
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/plain,hello",
        "javascript:alert(1)",
        "ftp://results.example.gov/file",
        "https://user:pass@results.example.gov/elections/2024",
        "https://127.0.0.1/elections/2024",
        "https://results.example.gov:8443/elections/2024",
    ],
)
def test_guard_rejects_unsafe_initial_url_shapes(url):
    resolver = resolver_with_public_hosts()
    with pytest.raises(PublicBrowserEgressError):
        PublicBrowserEgressGuard(
            approved_target_url=url,
            resolver=resolver,
        )


@pytest.mark.parametrize(
    "blocked_ip",
    [
        "127.0.0.1",
        "10.0.0.4",
        "172.16.1.2",
        "192.168.1.4",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "224.0.0.1",
    ],
)
def test_guard_rejects_private_loopback_linklocal_reserved_multicast_dns(
    blocked_ip,
):
    resolver = RecordingResolver(
        {"results.example.gov": (blocked_ip,)}
    )
    with pytest.raises(PublicBrowserEgressError):
        PublicBrowserEgressGuard(
            approved_target_url=APPROVED,
            resolver=resolver,
        )


def test_every_allowed_network_request_revalidates_dns():
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=resolver,
    )
    constructor_calls = len(resolver.calls)

    first = guard.evaluate_request(
        url=APPROVED,
        resource_type="document",
        top_level_navigation=True,
    )
    second = guard.evaluate_request(
        url="https://cdn.example.net/app.js",
        resource_type="script",
        top_level_navigation=False,
    )

    assert first.allowed is True
    assert second.allowed is True
    assert len(resolver.calls) == constructor_calls + 2
    assert resolver.calls[-2:] == [
        "results.example.gov",
        "cdn.example.net",
    ]


def test_cross_host_subresource_may_use_public_internet_but_top_level_may_not():
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=resolver,
    )

    assert guard.evaluate_request(
        url=APPROVED,
        resource_type="document",
        top_level_navigation=True,
    ).allowed is True

    subresource = guard.evaluate_request(
        url="https://cdn.example.net/app.js",
        resource_type="script",
        top_level_navigation=False,
    )
    assert subresource.allowed is True

    navigation = guard.evaluate_request(
        url="https://cdn.example.net/redirect",
        resource_type="document",
        top_level_navigation=True,
    )
    assert navigation.allowed is False
    assert navigation.reason == "top_level_cross_host_blocked"


@pytest.mark.parametrize(
    "resource_type",
    sorted(PUBLIC_BLOCKED_RESOURCE_TYPES),
)
def test_images_media_and_fonts_are_blocked_before_network(resource_type):
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=resolver,
    )
    before = len(resolver.calls)

    decision = guard.evaluate_request(
        url="https://cdn.example.net/asset",
        resource_type=resource_type,
        top_level_navigation=False,
    )

    assert decision.allowed is False
    assert decision.reason == "resource_type_blocked"
    assert len(resolver.calls) == before


def test_network_request_budget_is_hard_and_counts_attempts():
    policy = replace(
        DEFAULT_PUBLIC_RUN_POLICY,
        browser_network_request_max=2,
    )
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        policy=policy,
        resolver=resolver,
    )

    assert guard.evaluate_request(
        url=APPROVED,
        resource_type="document",
        top_level_navigation=True,
    ).allowed is True

    assert guard.evaluate_request(
        url="https://cdn.example.net/app.js",
        resource_type="script",
        top_level_navigation=False,
    ).allowed is True

    third = guard.evaluate_request(
        url="https://cdn.example.net/app.css",
        resource_type="stylesheet",
        top_level_navigation=False,
    )
    assert third.allowed is False
    assert third.reason == "network_request_budget_exceeded"
    assert guard.request_count == 3


def test_same_host_redirects_are_bounded():
    policy = replace(
        DEFAULT_PUBLIC_RUN_POLICY,
        top_level_redirect_max=2,
    )
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        policy=policy,
        resolver=resolver,
    )

    assert guard.evaluate_request(
        url=APPROVED,
        resource_type="document",
        top_level_navigation=True,
    ).allowed is True

    assert guard.evaluate_request(
        url="https://results.example.gov/redirect/one",
        resource_type="document",
        top_level_navigation=True,
    ).allowed is True

    assert guard.evaluate_request(
        url="https://results.example.gov/redirect/two",
        resource_type="document",
        top_level_navigation=True,
    ).allowed is True

    blocked = guard.evaluate_request(
        url="https://results.example.gov/redirect/three",
        resource_type="document",
        top_level_navigation=True,
    )
    assert blocked.allowed is False
    assert blocked.reason == "top_level_redirect_limit_exceeded"


def test_dns_rebinding_to_private_range_fails_closed_on_later_request():
    calls = {"count": 0}

    def rebinding_resolver(host):
        calls["count"] += 1
        if calls["count"] <= 2:
            return (PUBLIC_V4,)
        return ("169.254.169.254",)

    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=rebinding_resolver,
    )

    first = guard.evaluate_request(
        url=APPROVED,
        resource_type="document",
        top_level_navigation=True,
    )
    assert first.allowed is True

    second = guard.evaluate_request(
        url="https://results.example.gov/api/results",
        resource_type="xhr",
        top_level_navigation=False,
    )
    assert second.allowed is False
    assert second.reason.startswith("destination_blocked:")


def test_decision_does_not_expose_raw_url_or_resolved_ip():
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=resolver,
    )
    decision = guard.evaluate_request(
        url=APPROVED,
        resource_type="document",
        top_level_navigation=True,
    )
    rendered = repr(decision)
    assert APPROVED not in rendered
    assert PUBLIC_V4 not in rendered
    assert decision.destination_host_sha256 is not None


class FakeRequest:
    def __init__(self, url, resource_type, navigation):
        self.url = url
        self.resource_type = resource_type
        self._navigation = navigation

    def is_navigation_request(self):
        return self._navigation


class FakeRoute:
    def __init__(self):
        self.action = None
        self.code = None

    def continue_(self):
        self.action = "continue"

    def abort(self, code):
        self.action = "abort"
        self.code = code


def test_sync_playwright_compatible_route_handler_aborts_blocked_request():
    resolver = resolver_with_public_hosts()
    guard = PublicBrowserEgressGuard(
        approved_target_url=APPROVED,
        resolver=resolver,
    )
    route = FakeRoute()
    request = FakeRequest(
        "https://169.254.169.254/latest/meta-data",
        "xhr",
        False,
    )
    guard.handle_sync_route(route, request)
    assert route.action == "abort"
    assert route.code == "blockedbyclient"
