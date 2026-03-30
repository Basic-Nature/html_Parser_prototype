/* eslint-env jest */

describe('ballot_lens_modern sidebar hooks', () => {
  function loadScript() {
    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'ballot_lens_modern.js');
    const src = fs.readFileSync(filePath, 'utf8');
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
  }

  beforeAll(() => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: 640,
    });

    document.body.innerHTML = [
      '<div id="sidebar" class="sidebar-left"></div>',
      '<div class="sidebar-right"></div>',
      '<button id="sidebarToggleBtn" type="button"></button>',
      '<button id="btnToggleRightSidebar" type="button"></button>',
      '<div id="mobileSidebarOverlay" aria-hidden="true"></div>',
      '<input id="outputBypass" type="checkbox" />',
      '<button id="btnCancel" type="button"></button>',
      '<input id="searchResults" type="text" />',
      '<input id="filterConfidence" type="range" value="0" />',
      '<select id="filterState"></select>',
      '<select id="filterLevel"></select>',
      '<div id="logsList"></div>',
      '<div id="resultsGrid"></div>',
      '<div id="emptyState" class="hidden"></div>',
      '<div id="inputArtifacts"></div>'
    ].join('');

    const socketMock = {
      connected: true,
      id: 'socket-test',
      emit: jest.fn(),
      on: jest.fn(),
      off: jest.fn(),
      once: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      onevent: jest.fn(),
    };

    global.io = jest.fn(() => socketMock);
    window.io = global.io;

    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: async () => ({}),
      text: async () => '{}',
    });
    window.fetch = global.fetch;

    loadScript();
    document.dispatchEvent(new Event('DOMContentLoaded'));
  });

  test('exposes sidebar helpers on window', () => {
    expect(typeof window.openLeft).toBe('function');
    expect(typeof window.openRight).toBe('function');
    expect(typeof window.closeAll).toBe('function');
    expect(typeof window.setOverlayVisible).toBe('function');
  });

  test('openRight applies sidebar and overlay classes on mobile', () => {
    const rightSidebar = document.querySelector('.sidebar-right');
    const overlay = document.getElementById('mobileSidebarOverlay');

    window.openRight();

    expect(rightSidebar.classList.contains('open')).toBe(true);
    expect(rightSidebar.classList.contains('sidebar-open')).toBe(true);
    expect(overlay.classList.contains('visible')).toBe(true);
    expect(document.body.classList.contains('sidebar-right-open')).toBe(true);
  });

  test('closeAll clears sidebar and overlay classes', () => {
    const rightSidebar = document.querySelector('.sidebar-right');
    const overlay = document.getElementById('mobileSidebarOverlay');

    window.openRight();
    window.closeAll();

    expect(rightSidebar.classList.contains('open')).toBe(false);
    expect(overlay.classList.contains('visible')).toBe(false);
    expect(document.body.classList.contains('sidebar-right-open')).toBe(false);
  });

  test('setOverlayVisible keeps overlay hidden on desktop widths', () => {
    const overlay = document.getElementById('mobileSidebarOverlay');
    window.innerWidth = 1280;

    window.setOverlayVisible(true);

    expect(overlay.classList.contains('visible')).toBe(false);
    expect(overlay.getAttribute('aria-hidden')).toBe('true');
    expect(document.body.classList.contains('no-scroll')).toBe(false);
  });
});
