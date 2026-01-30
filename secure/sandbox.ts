/**
 * Moltbot Secure - Sandbox Execution
 *
 * Isolated Docker container for code/script execution.
 * Security-first: no network, read-only root, resource limits.
 */

import { spawn } from "node:child_process";
import type { SecureConfig } from "./config.js";
import type { AuditLogger } from "./audit.js";

export type SandboxResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  durationMs: number;
};

export type SandboxRunner = {
  run: (command: string, stdin?: string) => Promise<SandboxResult>;
  isAvailable: () => Promise<boolean>;
};

/**
 * Check if Docker is available
 */
async function checkDocker(): Promise<boolean> {
  return new Promise((resolve) => {
    const proc = spawn("docker", ["version"], {
      stdio: ["ignore", "ignore", "ignore"],
    });
    proc.on("error", () => resolve(false));
    proc.on("close", (code) => resolve(code === 0));
  });
}

/**
 * Build Docker run arguments for secure execution
 */
function buildDockerArgs(config: SecureConfig["sandbox"], command: string): string[] {
  const args: string[] = [
    "run",
    "--rm", // Remove container after exit
    "-i", // Interactive (for stdin)

    // Security: No network by default
    `--network=${config.network}`,

    // Security: Read-only root filesystem
    "--read-only",

    // Security: tmpfs for writable areas
    "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
    "--tmpfs=/var/tmp:rw,noexec,nosuid,size=64m",

    // Security: Drop all capabilities
    "--cap-drop=ALL",

    // Security: No new privileges
    "--security-opt=no-new-privileges",

    // Resource limits
    `--memory=${config.memory}`,
    `--cpus=${config.cpus}`,
    "--pids-limit=100",

    // Timeout handled externally, but set a ulimit too
    "--ulimit=cpu=60:60",

    // Working directory
    "--workdir=/workspace",

    // Image
    config.image,

    // Command (via shell for flexibility)
    "sh",
    "-c",
    command,
  ];

  return args;
}

export function createSandboxRunner(config: SecureConfig, audit: AuditLogger): SandboxRunner {
  const sandboxConfig = config.sandbox;

  return {
    async isAvailable(): Promise<boolean> {
      if (!sandboxConfig.enabled) return false;
      return checkDocker();
    },

    async run(command: string, stdin?: string): Promise<SandboxResult> {
      const startTime = Date.now();

      if (!sandboxConfig.enabled) {
        return {
          exitCode: 1,
          stdout: "",
          stderr: "Sandbox is disabled",
          timedOut: false,
          durationMs: 0,
        };
      }

      return new Promise((resolve) => {
        const args = buildDockerArgs(sandboxConfig, command);

        const proc = spawn("docker", args, {
          stdio: ["pipe", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";
        let timedOut = false;
        let resolved = false;

        const finish = (exitCode: number) => {
          if (resolved) return;
          resolved = true;

          const durationMs = Date.now() - startTime;

          audit.sandbox({
            command,
            exitCode,
            durationMs,
          });

          resolve({
            exitCode,
            stdout: stdout.slice(0, 10000), // Limit output size
            stderr: stderr.slice(0, 10000),
            timedOut,
            durationMs,
          });
        };

        // Timeout
        const timeout = setTimeout(() => {
          timedOut = true;
          proc.kill("SIGKILL");
        }, sandboxConfig.timeoutMs);

        proc.stdout?.on("data", (data: Buffer) => {
          stdout += data.toString();
          // Prevent memory exhaustion
          if (stdout.length > 100000) {
            proc.kill("SIGKILL");
          }
        });

        proc.stderr?.on("data", (data: Buffer) => {
          stderr += data.toString();
          if (stderr.length > 100000) {
            proc.kill("SIGKILL");
          }
        });

        proc.on("error", (err) => {
          clearTimeout(timeout);
          stderr += `\nProcess error: ${err.message}`;
          finish(1);
        });

        proc.on("close", (code) => {
          clearTimeout(timeout);
          finish(code ?? 1);
        });

        // Write stdin if provided
        if (stdin && proc.stdin) {
          proc.stdin.write(stdin);
          proc.stdin.end();
        } else {
          proc.stdin?.end();
        }
      });
    },
  };
}

/**
 * Parse sandbox command from user message
 * Returns null if message doesn't request code execution
 */
export function parseSandboxRequest(text: string): {
  language: string;
  code: string;
} | null {
  // Match code blocks with language
  const codeBlockMatch = text.match(/```(\w+)?\n([\s\S]*?)```/);
  if (codeBlockMatch) {
    const language = codeBlockMatch[1] || "sh";
    const code = codeBlockMatch[2].trim();
    return { language, code };
  }

  // Match /run command
  const runMatch = text.match(/^\/run\s+(.+)$/s);
  if (runMatch) {
    return { language: "sh", code: runMatch[1].trim() };
  }

  // Match /python command
  const pythonMatch = text.match(/^\/python\s+(.+)$/s);
  if (pythonMatch) {
    return { language: "python", code: pythonMatch[1].trim() };
  }

  return null;
}

/**
 * Build execution command for language
 */
export function buildCommand(language: string, code: string): string {
  switch (language.toLowerCase()) {
    case "python":
    case "py":
      // Write code to temp file and execute
      return `python3 -c ${JSON.stringify(code)}`;

    case "javascript":
    case "js":
    case "node":
      return `node -e ${JSON.stringify(code)}`;

    case "bash":
    case "sh":
    case "shell":
      return code;

    default:
      // Default to shell
      return code;
  }
}

/**
 * Format sandbox result for display
 */
export function formatSandboxResult(result: SandboxResult): string {
  let output = "";

  if (result.timedOut) {
    output += "**Timed out**\n\n";
  }

  if (result.stdout) {
    output += "**Output:**\n```\n" + result.stdout.trim() + "\n```\n";
  }

  if (result.stderr) {
    output += "**Errors:**\n```\n" + result.stderr.trim() + "\n```\n";
  }

  if (!result.stdout && !result.stderr) {
    output += result.exitCode === 0 ? "Command completed (no output)" : "Command failed (no output)";
  }

  output += `\n_Exit code: ${result.exitCode}, Duration: ${result.durationMs}ms_`;

  return output;
}
