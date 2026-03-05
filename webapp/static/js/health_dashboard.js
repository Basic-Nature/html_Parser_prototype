(() => {
  const configEl = document.getElementById("healthTaskConfig");
  if (!configEl) {
      return;
  }

  const tableBody = document.querySelector("[data-task-status]");
  const taskButtons = document.querySelectorAll("[data-task-key]");
  const refreshButton = document.querySelector("[data-refresh-health-tasks]");
  const socketTestButton = document.querySelector("[data-socket-test-broadcast]");
  const alertBox = document.querySelector("[data-health-alert]");
  const defaultAlert = alertBox ? alertBox.textContent.trim() : "";
  const healthEnabled = String(configEl.dataset.healthEnabled || "true").toLowerCase() === "true";
  const healthReason = String(configEl.dataset.healthReason || "").trim();

  const parseJson = (value) => {
      if (!value) {
          return [];
      }
      try {
          return JSON.parse(value);
      } catch (err) {
          console.warn("Failed to parse JSON payload", err);
          return [];
      }
  };

  const renderTasks = (tasks) => {
      if (!tableBody) {
          return;
      }
      tableBody.innerHTML = "";
      if (!tasks || !tasks.length) {
          const row = document.createElement("tr");
          const cell = document.createElement("td");
          cell.colSpan = 4;
          cell.className = "text-center text-muted";
          cell.textContent = "No tasks queued yet. Launch one above.";
          row.appendChild(cell);
          tableBody.appendChild(row);
          return;
      }
      tasks.forEach((task) => {
          const row = document.createElement("tr");
          row.dataset.taskRow = task.id;

          const titleCell = document.createElement("td");
          const title = document.createElement("div");
          title.className = "fw-semibold";
          title.textContent = task.label || task.task;
          const subtitle = document.createElement("div");
          subtitle.className = "small text-muted";
          subtitle.textContent = task.description || "Python module";
          titleCell.appendChild(title);
          titleCell.appendChild(subtitle);

          const statusCell = document.createElement("td");
          const badge = document.createElement("span");
          badge.className = `badge ${statusClass(task.status)}`;
          badge.textContent = (task.status || "queued").charAt(0).toUpperCase() + (task.status || "queued").slice(1);
          statusCell.appendChild(badge);

          const timingCell = document.createElement("td");
          timingCell.className = "small";
          const start = document.createElement("div");
          const startLabel = document.createElement("strong");
          startLabel.textContent = "Start:";
          start.appendChild(startLabel);
          start.append(" ", task.started_at || "—");
          const end = document.createElement("div");
          const endLabel = document.createElement("strong");
          endLabel.textContent = "End:";
          end.appendChild(endLabel);
          end.append(" ", task.ended_at || "in progress");
          timingCell.appendChild(start);
          timingCell.appendChild(end);

          const logCell = document.createElement("td");
          logCell.className = "small";
          const details = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = "View log";
          const pre = document.createElement("pre");
          pre.className = "mb-0";
          pre.textContent = task.log || "No output yet.";
          details.appendChild(summary);
          details.appendChild(pre);
          logCell.appendChild(details);

          row.appendChild(titleCell);
          row.appendChild(statusCell);
          row.appendChild(timingCell);
          row.appendChild(logCell);
          tableBody.appendChild(row);
      });
  };

  const statusClass = (status) => {
      switch ((status || "queued").toLowerCase()) {
          case "completed":
              return "bg-success";
          case "failed":
              return "bg-danger";
          case "running":
              return "bg-secondary";
          default:
              return "bg-info";
      }
  };

  const pushAlert = (message, variant = "info") => {
      if (!alertBox) {
          return;
      }
      alertBox.textContent = message || defaultAlert;
      alertBox.className = `alert alert-${variant} shadow-sm`;
  };

  const startTask = async (button) => {
      if (!healthEnabled) {
          pushAlert("Health task execution is disabled in this environment.", "warning");
          return;
      }
      const key = button.dataset.taskKey;
      if (!key) {
          return;
      }
      const originalLabel = button.textContent;
      button.disabled = true;
      button.textContent = "Launching...";
      try {
          // Use AuthUtils for de-duplication if available
          const winAny = (typeof window !== 'undefined') ? /** @type {any} */ (window) : null;
          const authUtils = winAny ? winAny.AuthUtils : null;
          
          const mutationKey = `health_task:${key}`;
          const execMutation = async () => {
            return fetch("/api/health_tasks", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ task: key })
            });
          };
          
          const resp = authUtils && typeof authUtils.executeMutationOnce === 'function'
            ? await authUtils.executeMutationOnce(mutationKey, execMutation)
            : await execMutation();
          
          const data = await resp.json();
          if (!resp.ok) {
              throw new Error(data.error || "Failed to start task.");
          }
          pushAlert(`Started ${data.task?.label || key}`, "success");
          await fetchTasks(false);
      } catch (error) {
          pushAlert(error.message || "Failed to start task.", "danger");
      } finally {
          button.disabled = false;
          button.textContent = originalLabel;
      }
  };

  const fetchTasks = async (announce = false) => {
      if (!healthEnabled) {
          renderTasks(parseJson(configEl.dataset.initial));
          if (announce) {
              pushAlert("Read-only mode: task execution is disabled.", "warning");
          }
          return;
      }
      try {
          const resp = await fetch("/api/health_tasks");
          if (!resp.ok) {
              throw new Error("Unable to read task history.");
          }
          const data = await resp.json();
          renderTasks(data.tasks || []);
          if (announce) {
              pushAlert("Task list refreshed.", "info");
          }
      } catch (error) {
          pushAlert(error.message || "Unable to refresh tasks.", "warning");
      }
  };

  // Wire up buttons
  taskButtons.forEach((button) => {
      button.addEventListener("click", () => startTask(button));
  });
  if (refreshButton) {
      refreshButton.addEventListener("click", () => fetchTasks(true));
  }
  if (socketTestButton instanceof HTMLButtonElement) {
      socketTestButton.addEventListener("click", async () => {
          if (!healthEnabled) {
              pushAlert("Read-only mode: socket test broadcast is disabled.", "warning");
              return;
          }
          socketTestButton.disabled = true;
          try {
              const winAny = (typeof window !== 'undefined') ? /** @type {any} */ (window) : null;
              const authUtils = winAny ? winAny.AuthUtils : null;
              const execMutation = async () => {
                  return fetch("/api/health_socket_test", { method: "POST" });
              };
              const resp = authUtils && typeof authUtils.executeMutationOnce === 'function'
                  ? await authUtils.executeMutationOnce("health_socket_test", execMutation)
                  : await execMutation();
              const data = await resp.json().catch(() => null);
              if (!resp.ok) {
                  throw new Error(data?.error || "Unable to broadcast socket test.");
              }
              const instance = data?.payload?.instance_id || data?.payload?.hostname || "unknown";
              pushAlert(`Socket test broadcast sent from ${instance}.`, "info");
          } catch (error) {
              pushAlert(error.message || "Unable to broadcast socket test.", "danger");
          } finally {
              socketTestButton.disabled = false;
          }
      });
  }

  // Render any initial payload embedded by the server.
  renderTasks(parseJson(configEl.dataset.initial));

  if (!healthEnabled) {
      if (healthReason === "health_tasks_disabled") {
          pushAlert("Read-only mode: health tasks are disabled by policy.", "warning");
      } else if (healthReason === "health_token_missing") {
          pushAlert("Read-only mode: HEALTH_TASK_TOKEN is missing in environment configuration.", "warning");
      }
      return;
  }

  // Begin polling every 7 seconds for updates.
  fetchTasks(false);
  const pollInterval = 7000;
  setInterval(fetchTasks, pollInterval);
})();
