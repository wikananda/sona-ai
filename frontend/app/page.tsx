"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import NewProjectForm from "@/src/components/NewProjectForm";
import { createProject, deleteProject, listProjects, Project } from "@/src/api/sonaApi";

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState("");

  const refreshProjects = async () => {
    setError("");
    const data = await listProjects();
    setProjects(data);
  };

  useEffect(() => {
    refreshProjects()
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, []);

  const handleCreate = async (params: { name: string; description?: string }) => {
    setIsCreating(true);
    setError("");
    try {
      const project = await createProject(params);
      setProjects((current) => [project, ...current]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (projectId: string) => {
    if (!window.confirm("Delete this project and all recordings?")) return;

    setError("");
    try {
      await deleteProject(projectId);
      setProjects((current) => current.filter((project) => project.id !== projectId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete project");
    }
  };

  return (
    <main className="min-h-screen bg-zinc-100 text-zinc-950">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-5 py-8">
        <header className="flex flex-col gap-1">
          <h1 className="text-2xl font-semibold">Sona AI</h1>
          <p className="text-sm text-zinc-600">Projects</p>
        </header>

        <NewProjectForm onCreate={handleCreate} isCreating={isCreating} />

        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        )}

        <section className="rounded-lg border border-zinc-200 bg-white">
          <div className="border-b border-zinc-200 px-4 py-3">
            <h2 className="text-sm font-semibold">Recent projects</h2>
          </div>

          {isLoading && (
            <div className="px-4 py-8 text-sm text-zinc-500">Loading projects...</div>
          )}

          {!isLoading && projects.length === 0 && (
            <div className="px-4 py-10 text-sm text-zinc-500">
              Create a project to start uploading recordings.
            </div>
          )}

          <div className="divide-y divide-zinc-200">
            {projects.map((project) => (
              <div key={project.id} className="flex items-center justify-between gap-4 px-4 py-4">
                <Link href={`/projects/${project.id}`} className="min-w-0 flex-1">
                  <h3 className="truncate text-base font-medium text-zinc-950">
                    {project.name}
                  </h3>
                  <p className="mt-1 truncate text-sm text-zinc-500">
                    {project.description || "No description"}
                  </p>
                </Link>
                <div className="flex items-center gap-4">
                  <span className="hidden text-xs text-zinc-500 sm:inline">
                    {formatDate(project.created_at)}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleDelete(project.id)}
                    className="rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-700 hover:border-red-300 hover:text-red-700"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}
