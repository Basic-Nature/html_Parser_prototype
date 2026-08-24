(function(root, factory) {
  'use strict';

  const api = factory();

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }

  if (root && typeof root === 'object') {
    root.PipelineInspectionPanel = api;
  }
})(
  typeof window !== 'undefined'
    ? window
    : (typeof globalThis !== 'undefined' ? globalThis : null),
  function() {
    'use strict';

    const CUSTOM_EVENT = 'pipeline:inspection';
    const ENVELOPE_CONTRACT = 'pipeline_inspection_socket_v1';
    const INSPECTION_CONTRACT = 'pipeline_inspection_v1';

    function isPlainObject(value) {
      return !!value
        && typeof value === 'object'
        && !Array.isArray(value);
    }

    function nullableText(value) {
      if (value === null || typeof value === 'undefined') {
        return 'Not provided';
      }
      return String(value);
    }

    function booleanText(value) {
      if (value === true) {
        return 'Yes';
      }
      if (value === false) {
        return 'No';
      }
      return 'Not provided';
    }

    function buildTransformationView(item) {
      const source = isPlainObject(item) ? item : {};
      const details = isPlainObject(source.details) ? source.details : {};

      return {
        sequence: Number.isInteger(source.sequence)
          ? source.sequence
          : null,
        operation: typeof source.operation === 'string'
          ? source.operation
          : null,
        fromStage: typeof source.from_stage === 'string'
          ? source.from_stage
          : null,
        toStage: typeof source.to_stage === 'string'
          ? source.to_stage
          : null,
        ruleSource: typeof source.rule_source === 'string'
          ? source.rule_source
          : null,
        confidence: (
          typeof source.confidence === 'number'
          && Number.isFinite(source.confidence)
        )
          ? source.confidence
          : null,
        evidenceRefCount: Array.isArray(source.evidence_refs)
          ? source.evidence_refs.length
          : 0,
        beforeHeader: typeof details.before_header === 'string'
          ? details.before_header
          : null,
        afterHeader: typeof details.after_header === 'string'
          ? details.after_header
          : null,
        voteValueMutation: typeof details.vote_value_mutation === 'boolean'
          ? details.vote_value_mutation
          : null,
        semanticValueMutation: (
          typeof details.semantic_value_mutation === 'boolean'
        )
          ? details.semantic_value_mutation
          : null,
      };
    }

    function buildWarningView(item) {
      const source = isPlainObject(item) ? item : {};

      return {
        severity: typeof source.severity === 'string'
          ? source.severity
          : null,
        code: typeof source.code === 'string'
          ? source.code
          : null,
        message: typeof source.message === 'string'
          ? source.message
          : null,
      };
    }

    function buildViewModel(envelope) {
      if (!isPlainObject(envelope)) {
        throw new TypeError('Why panel requires an inspection envelope');
      }
      if (envelope.contract !== ENVELOPE_CONTRACT) {
        throw new Error('Unexpected inspection envelope contract');
      }
      if (
        !isPlainObject(envelope.authority)
        || envelope.authority.canonical !== false
      ) {
        throw new Error('Why panel accepts noncanonical evidence only');
      }

      const inspection = envelope.inspection;
      if (
        !isPlainObject(inspection)
        || inspection.contract !== INSPECTION_CONTRACT
      ) {
        throw new Error('Unexpected inspection payload contract');
      }
      if (
        !isPlainObject(inspection.authority)
        || inspection.authority.canonical !== false
      ) {
        throw new Error('Inspection payload must be noncanonical');
      }
      if (
        inspection.rows_included !== false
        || inspection.headers_included !== false
      ) {
        throw new Error('Why panel refuses raw rows or headers');
      }

      const summary = isPlainObject(inspection.summary)
        ? inspection.summary
        : {};
      const completeness = isPlainObject(inspection.completeness)
        ? inspection.completeness
        : {};
      const provenance = isPlainObject(inspection.source_provenance)
        ? inspection.source_provenance
        : {};

      if (
        provenance.source_uri_included !== false
        || provenance.source_metadata_included !== false
      ) {
        throw new Error('Why panel refuses source URI or source metadata');
      }

      const transformations = Array.isArray(inspection.transformations)
        ? inspection.transformations.map(buildTransformationView)
        : [];
      const warnings = Array.isArray(inspection.warnings)
        ? inspection.warnings.map(buildWarningView)
        : [];

      return {
        sessionId: typeof envelope.session_id === 'string'
          ? envelope.session_id
          : null,
        stage: typeof inspection.stage === 'string'
          ? inspection.stage
          : null,
        authorityLabel: 'NONE',
        authorityExplanation: (
          'Noncanonical parser evidence only. '
          + 'This view cannot approve or promote election data.'
        ),
        sourceType: typeof provenance.source_type === 'string'
          ? provenance.source_type
          : null,
        evidenceRef: typeof provenance.evidence_ref === 'string'
          ? provenance.evidence_ref
          : null,
        artifactId: typeof provenance.artifact_id === 'string'
          ? provenance.artifact_id
          : null,
        rowCount: Number.isInteger(summary.row_count)
          ? summary.row_count
          : null,
        headerCount: Number.isInteger(summary.header_count)
          ? summary.header_count
          : null,
        transformationCount: Number.isInteger(summary.transformation_count)
          ? summary.transformation_count
          : transformations.length,
        warningCount: Number.isInteger(summary.warning_count)
          ? summary.warning_count
          : warnings.length,
        completeness: {
          state: typeof completeness.state === 'string'
            ? completeness.state
            : null,
          expectedCount: Number.isInteger(completeness.expected_count)
            ? completeness.expected_count
            : null,
          observedCount: Number.isInteger(completeness.observed_count)
            ? completeness.observed_count
            : null,
          missingCount: Number.isInteger(completeness.missing_count)
            ? completeness.missing_count
            : null,
          nullValueCount: Number.isInteger(completeness.null_value_count)
            ? completeness.null_value_count
            : null,
          isComplete: typeof completeness.is_complete === 'boolean'
            ? completeness.is_complete
            : null,
        },
        transformations,
        warnings,
      };
    }

    function appendDefinition(documentRef, list, term, value) {
      const dt = documentRef.createElement('dt');
      dt.textContent = term;

      const dd = documentRef.createElement('dd');
      dd.textContent = nullableText(value);

      list.appendChild(dt);
      list.appendChild(dd);
    }

    function appendTransformation(documentRef, host, item, index) {
      const card = documentRef.createElement('article');
      card.className = 'source-card';
      card.setAttribute(
        'aria-label',
        `Transformation ${index + 1}`
      );

      const header = documentRef.createElement('div');
      header.className = 'source-card-header';

      const title = documentRef.createElement('div');
      title.className = 'source-title';
      title.textContent = item.operation || `Transformation ${index + 1}`;
      header.appendChild(title);
      card.appendChild(header);

      const body = documentRef.createElement('div');
      body.className = 'source-body';

      const list = documentRef.createElement('dl');
      list.className = 'info-list';

      if (item.beforeHeader !== null) {
        appendDefinition(
          documentRef,
          list,
          'Observed',
          item.beforeHeader
        );
      }
      if (item.afterHeader !== null) {
        appendDefinition(
          documentRef,
          list,
          'Interpreted',
          item.afterHeader
        );
      }

      appendDefinition(
        documentRef,
        list,
        'From stage',
        item.fromStage
      );
      appendDefinition(
        documentRef,
        list,
        'To stage',
        item.toStage
      );
      appendDefinition(
        documentRef,
        list,
        'Rule source',
        item.ruleSource
      );
      appendDefinition(
        documentRef,
        list,
        'Confidence',
        item.confidence
      );

      const mutationValue = item.voteValueMutation !== null
        ? item.voteValueMutation
        : item.semanticValueMutation;

      appendDefinition(
        documentRef,
        list,
        'Vote values changed',
        mutationValue === null
          ? null
          : booleanText(mutationValue)
      );
      appendDefinition(
        documentRef,
        list,
        'Evidence references',
        item.evidenceRefCount
      );

      body.appendChild(list);
      card.appendChild(body);
      host.appendChild(card);
    }

    function appendWarning(documentRef, host, item, index) {
      const card = documentRef.createElement('article');
      card.className = 'source-card';
      card.setAttribute('aria-label', `Warning ${index + 1}`);

      const header = documentRef.createElement('div');
      header.className = 'source-card-header';

      const title = documentRef.createElement('div');
      title.className = 'source-title';
      title.textContent = item.code || `Warning ${index + 1}`;
      header.appendChild(title);
      card.appendChild(header);

      const body = documentRef.createElement('div');
      body.className = 'source-body';

      const list = documentRef.createElement('dl');
      list.className = 'info-list';
      appendDefinition(
        documentRef,
        list,
        'Severity',
        item.severity
      );
      appendDefinition(
        documentRef,
        list,
        'Message',
        item.message
      );

      body.appendChild(list);
      card.appendChild(body);
      host.appendChild(card);
    }

    function renderViewModel(viewModel, documentRef) {
      const panel = documentRef.getElementById('pipelineInspectionPanel');
      const badge = documentRef.getElementById('pipelineInspectionBadge');
      const stagePreview = documentRef.getElementById(
        'pipelineInspectionStagePreview'
      );
      const summaryHost = documentRef.getElementById(
        'pipelineInspectionSummary'
      );
      const transformationsHost = documentRef.getElementById(
        'pipelineInspectionTransformations'
      );
      const warningsHost = documentRef.getElementById(
        'pipelineInspectionWarnings'
      );

      if (
        !panel
        || !badge
        || !stagePreview
        || !summaryHost
        || !transformationsHost
        || !warningsHost
      ) {
        return false;
      }

      const wasHidden = panel.hidden === true;

      badge.textContent = 'noncanonical';
      stagePreview.textContent = viewModel.stage
        ? viewModel.stage.toUpperCase()
        : 'UNKNOWN';

      summaryHost.replaceChildren();

      const summaryList = documentRef.createElement('dl');
      summaryList.className = 'info-list';

      appendDefinition(
        documentRef,
        summaryList,
        'Stage',
        viewModel.stage
      );
      appendDefinition(
        documentRef,
        summaryList,
        'Source type',
        viewModel.sourceType
      );
      appendDefinition(
        documentRef,
        summaryList,
        'Rows observed',
        viewModel.rowCount
      );
      appendDefinition(
        documentRef,
        summaryList,
        'Headers observed',
        viewModel.headerCount
      );
      appendDefinition(
        documentRef,
        summaryList,
        'Completeness',
        viewModel.completeness.state
      );
      appendDefinition(
        documentRef,
        summaryList,
        'Canonical authority',
        viewModel.authorityLabel
      );

      const authorityNote = documentRef.createElement('p');
      authorityNote.className = 'artifact-copy';
      authorityNote.textContent = viewModel.authorityExplanation;

      summaryHost.appendChild(summaryList);
      summaryHost.appendChild(authorityNote);

      transformationsHost.replaceChildren();
      if (viewModel.transformations.length === 0) {
        const empty = documentRef.createElement('p');
        empty.className = 'artifact-copy';
        empty.textContent = (
          'No transformation records were included in this inspection payload.'
        );
        transformationsHost.appendChild(empty);
      }
      else {
        viewModel.transformations.forEach((item, index) => {
          appendTransformation(
            documentRef,
            transformationsHost,
            item,
            index
          );
        });
      }

      warningsHost.replaceChildren();
      if (viewModel.warnings.length === 0) {
        const empty = documentRef.createElement('p');
        empty.className = 'artifact-copy';
        empty.textContent = 'No parser warnings in this inspection payload.';
        warningsHost.appendChild(empty);
      }
      else {
        viewModel.warnings.forEach((item, index) => {
          appendWarning(documentRef, warningsHost, item, index);
        });
      }

      panel.hidden = false;
      if (wasHidden && 'open' in panel) {
        panel.open = true;
      }

      return true;
    }

    function handleInspectionEvent(event, documentRef) {
      if (!event || !isPlainObject(event.detail)) {
        return false;
      }

      const viewModel = buildViewModel(event.detail);
      return renderViewModel(viewModel, documentRef);
    }

    function attach(documentRef) {
      if (
        !documentRef
        || typeof documentRef.addEventListener !== 'function'
      ) {
        throw new TypeError('Why panel requires a document event target');
      }

      const handler = function(event) {
        try {
          handleInspectionEvent(event, documentRef);
        }
        catch (error) {
          console.warn(
            '[Pipeline Inspection Panel] Rejected explanation payload:',
            error && error.message ? error.message : 'unknown_error'
          );
        }
      };

      documentRef.addEventListener(CUSTOM_EVENT, handler);

      return function detach() {
        documentRef.removeEventListener(CUSTOM_EVENT, handler);
      };
    }

    if (
      typeof document !== 'undefined'
      && document
      && typeof document.addEventListener === 'function'
    ) {
      attach(document);
    }

    return Object.freeze({
      CUSTOM_EVENT,
      ENVELOPE_CONTRACT,
      INSPECTION_CONTRACT,
      buildViewModel,
      renderViewModel,
      handleInspectionEvent,
      attach,
    });
  }
);