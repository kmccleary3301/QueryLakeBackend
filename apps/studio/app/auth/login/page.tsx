"use client";

import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
// import { cookies } from 'next/headers' // Doesn't work with client components
// import Cookies from 'js-cookie';
import { setCookie } from "@/hooks/cookies";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useForm } from "react-hook-form"
import { SERVER_ADDR_HTTP } from "@/config_server_hostnames"
import craftUrl from "@/hooks/craftUrl"
import { userDataType } from "@/types/globalTypes";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form"
// import { toast } from "@/registry/default/ui/use-toast";
import { toast } from "sonner";
import { useContextAction } from "@/app/context-provider";
import { useRouter } from 'next/navigation';
import CompactInput from "@/components/ui/compact-input";


type login_results = {
  success: false,
  error: string
} | {
  success: true,
  result: userDataType
};

const formSchema = z.object({
  username: z.string().min(1, {
    message: "Username must be at least 1 character.",
  }),
	password: z.string().min(1, {
    message: "Password must be at least 1 character.",
  })
})


export default function LoginPage() {
  const { setUserData, getUserData } = useContextAction();
  const router = useRouter();
  // setTheme("light");

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
			password: ""
    },
  })

  const setUserDataHook = (data: userDataType) => {
    // Cookies.set("UD", JSON.stringify(data), { secure: true })
    setCookie({ key: "UD", value: data });
    setUserData(data);
    router.push("/select-workspace");
    return;
  }

  const setErrorMessage = (message: string) => {
    toast(message);
  }

  const login = (values: z.infer<typeof formSchema>) => {
    fetch(`/api/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        auth: {
          username: values.username,
          password: values.password,
        },
      }),
    }).then(async (response) => {
      response.json().then((data : login_results) => {
        console.log("GOT LOGIN DATA:", data);
				if (data.success) {
					const result : userDataType = data.result;
          getUserData(result.auth, () => {});
				} else {
					setErrorMessage(data.error as string);
				}
      });
    });
  }

  return (
    <div className="mx-auto space-y-2 max-w-sm">
			<div className="text-center space-y-0 pb-3">
        <h1 className="text-3xl font-bold">Log In</h1>
      </div>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(login)} className="space-y-6">
          <div className="space-y-2">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <CompactInput className="h-11 outline-none" backgroundColor="#09090B" placeholder="Username or Email" autoComplete="username" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="space-y-2">
            
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <CompactInput className="h-11" backgroundColor="#09090B" placeholder="Password" type="password" autoComplete="current-password" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="pb-2">
            <Button className="w-full h-8 text-black" type="submit">
              Login
            </Button>
          </div>
          {/* <Button variant="secondary" type="submit" style={{fontSize: 20}}>Submit</Button> */}
        </form>
      </Form>
      <Separator />
      <div className="space-y-2 pt-2">
        <p className="text-center text-sm flex flex-row justify-center">
          Don&apos;t have an account?
          <Link className="underline pl-2" href="/auth/sign_up">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  )
}
