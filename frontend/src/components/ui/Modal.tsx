import { ReactNode } from "react";

interface Props {
    children: ReactNode;
    zClassName?: string;
    backdropClassName?: string;
}

export default function Modal({
    children,
    zClassName = "z-50",
    backdropClassName = "bg-black/40",
}: Props) {
    return (
        <div className={`fixed inset-0 ${zClassName} flex items-center justify-center ${backdropClassName} px-4`}>
            {children}
        </div>
    );
}
