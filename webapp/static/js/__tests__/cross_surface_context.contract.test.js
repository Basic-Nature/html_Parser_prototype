const fs = require('fs');
const path = require('path');

const {
  CONTEXT_KEYS,
  TARGET_PATHS,
  buildContextHref,
} = require('../cross_surface_context.js');

describe('O4E cross-surface presentation context contract', () => {
  const origin = 'https://www.electionpulse.org';

  test('exports only the approved presentation context and known routes', () => {
    expect(CONTEXT_KEYS).toEqual(['state', 'year']);
    expect(TARGET_PATHS).toEqual(['/worklist', '/data_framework']);
  });

  test('copies state and year Data Framework -> Workflow and nothing else', () => {
    const href = buildContextHref(
      `${origin}/data_framework?state=TX&year=2024&registry_source_id=source-1&workflow_item_id=item-1&search=Senate`,
      '/worklist',
      '/worklist',
      origin
    );
    expect(href).toBe('/worklist?state=TX&year=2024');
    expect(href).not.toContain('registry_source_id');
    expect(href).not.toContain('workflow_item_id');
    expect(href).not.toContain('search=');
  });

  test('copies state and year Workflow -> Data Framework and nothing else', () => {
    const href = buildContextHref(
      `${origin}/worklist?state=AZ&year=2026&workflow_pass_id=pass-1&expected_row_version=8&contest=Mayor`,
      '/data_framework',
      '/data_framework',
      origin
    );
    expect(href).toBe('/data_framework?state=AZ&year=2026');
    expect(href).not.toContain('workflow_pass_id');
    expect(href).not.toContain('expected_row_version');
    expect(href).not.toContain('contest=');
  });

  test('omits absent or empty approved values', () => {
    expect(
      buildContextHref(
        `${origin}/data_framework?state=&year=`,
        '/worklist',
        '/worklist',
        origin
      )
    ).toBe('/worklist');
  });

  test('fails closed for a cross-origin target', () => {
    expect(
      buildContextHref(
        `${origin}/data_framework?state=TX&year=2024`,
        'https://example.invalid/worklist',
        '/worklist',
        origin
      )
    ).toBeNull();
  });

  test('fails closed for a source outside the declared origin', () => {
    expect(
      buildContextHref(
        'https://example.invalid/data_framework?state=TX&year=2024',
        '/worklist',
        '/worklist',
        origin
      )
    ).toBeNull();
  });

  test('fails closed when the marker path does not match the target path', () => {
    expect(
      buildContextHref(
        `${origin}/worklist?state=TX&year=2024`,
        '/data_framework',
        '/worklist',
        origin
      )
    ).toBeNull();
  });

  test('templates wire only the selected anchors and preserve direct href fallbacks', () => {
    const templates = path.join(__dirname, '..', '..', '..', 'templates');
    const dataFramework = fs.readFileSync(
      path.join(templates, 'data_framework.html'),
      'utf8'
    );
    const workflow = fs.readFileSync(
      path.join(templates, 'worklist.html'),
      'utf8'
    );

    expect(
      dataFramework.match(/data-o4e-context-nav="\/worklist"/g) || []
    ).toHaveLength(1);
    expect(
      workflow.match(/data-o4e-context-nav="\/data_framework"/g) || []
    ).toHaveLength(1);

    expect(dataFramework).toContain("filename='js/cross_surface_context.js'");
    expect(workflow).toContain("filename='js/cross_surface_context.js'");
    expect(dataFramework).toContain("{{ url_for('worklist') }}");
    expect(workflow).toContain("{{ url_for('data_framework') }}");

    const workflowAnchors = workflow.match(
      /<a\b[^>]*url_for\('data_framework'\)[^>]*>/g
    ) || [];
    expect(workflowAnchors).toHaveLength(2);

    const primary = workflowAnchors.find((tag) =>
      tag.includes('workflow-link-primary')
    );
    const appNav = workflowAnchors.find((tag) =>
      tag.includes('ep-app-nav__link')
    );

    expect(primary).toContain('data-o4e-context-nav="/data_framework"');
    expect(appNav).not.toContain('data-o4e-context-nav');
  });
});
